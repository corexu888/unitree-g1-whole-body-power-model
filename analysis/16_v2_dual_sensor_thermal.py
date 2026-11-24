"""v2 candidates (b) and (c), on top of the actuator-tied structure (g.ACT_GROUPS):
   (b) 10 Hz target = mainboard reading + per-boot offset to the BMS level (offset from the two sensors only, no model);
   (c) thermal creep: copper loss grows with the heat accumulated in the winding during the segment."""
import sys, os, json
import numpy as np, pandas as pd
from scipy.optimize import nnls
from scipy.signal import lfilter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1power as g
ROOT = sys.argv[1]; A = os.path.join(ROOT, "analysis")
pd.set_option("display.width", 250, "display.max_rows", 400, "display.max_columns", 60)
df = g.load(ROOT)
KW = dict(fixed={"mech_pos": 1.0}, upper={"mech_neg": 1.0})
tau, dq, T = g.arrays(df)
X16 = g.features(df, dict(cu="type", mech="split")); X16.insert(0, "const", 1.0)
X = g.tie_types(X16)
segv = df.segid.values; segi = df.groupby("segid", sort=False).cumcount().values
K = len(g.BMS_FIR) - 1

# ------------------------------------------------------------------ thermal creep regressors (per frame)
dt = df.groupby("segid").t.diff().fillna(0.103).clip(upper=0.3).values
t2 = tau ** 2
t2_0 = pd.DataFrame(np.where((segi < 15)[:, None], t2, np.nan)).groupby(segv).transform("mean").values      # tau^2 at the start of the segment
ex = (t2 - t2_0) * dt[:, None]
def heat(Tw):
    """integral of (tau^2 - tau0^2) dt within the segment, optionally leaking with time constant Tw [s]; unit (N m)^2 s"""
    S = np.zeros_like(ex)
    a = 1.0 if not np.isfinite(Tw) else float(np.exp(-0.103 / Tw))
    for sid, idx in df.groupby("segid", sort=False).indices.items():
        S[idx] = lfilter([1.0], [1.0, -a], ex[idx], axis=0)
    return S
def creep_cols(Tw):
    S = heat(Tw)
    return pd.DataFrame({"creep_arm": (t2[:, g.ARM] * S[:, g.ARM]).sum(1), "creep_leg": (t2[:, g.LEG] * S[:, g.LEG]).sum(1),
                         "creep_waist": (t2[:, g.WAIST] * S[:, g.WAIST]).sum(1)}, index=df.index)

# ------------------------------------------------------------------ BMS rows
Xc = pd.concat([X, creep_cols(np.inf).add_suffix("_inf"), creep_cols(60.0).add_suffix("_60"), creep_cols(20.0).add_suffix("_20")], axis=1)
meta_all, Xf_all = g.bms_align(df, Xc); kk = (meta_all.fsm != 1).values
meta, Xf = meta_all[kk], Xf_all[kk]; y = meta.P_bms.values
BASE = list(X.columns)
c_bms = g.fit_bounded(Xf[BASE], y, **KW)
p_bms_cv, C_bms = g.lobo_cv(meta, Xf[BASE], y, meta.boot.values, **KW)

# ------------------------------------------------------------------ (b) fused 10 Hz target
df["P_mb_f"] = df.groupby("segid").P_mb.transform(lambda s: np.convolve(s.values, g.BMS_FIR, mode="full")[:len(s)])
df.loc[segi < K, "P_mb_f"] = np.nan
fresh = df.groupby("segid").cur_raw.transform(lambda s: np.r_[False, np.diff(s.values) != 0]).values.astype(bool)
ok = fresh & df.P_mb_f.notna().values
off = (df.P_bms - df.P_mb_f)[ok].groupby(df.boot[ok]).mean()
df["P_fus"] = df.P_mb + df.boot.map(off)
print("per-boot offset BMS - mainboard (W):", off.round(2).to_dict())

def shift(v, k):
    """value k frames earlier within the same segment (k<0: later); NaN outside the segment"""
    s = pd.DataFrame(v).groupby(segv).shift(k).values
    return s
LAGS = [-1, 0, 1, 2, 3, 4]
p_inst = X[BASE].values @ c_bms[BASE].values
L = np.column_stack([shift(p_inst, kq)[:, 0] for kq in LAGS])
static = ~df.action.isin(g.LOCO).values
val = np.isfinite(L).all(1) & (df.fsm != 1).values & df.P_fus.notna().values
for nm, mm in [("all", val), ("arm/static", val & static), ("locomotion", val & ~static)]:
    w, _ = nnls(L[mm], df.P_fus.values[mm])
    print(f"mainboard response to telemetry, lags {LAGS} frames, {nm}: weights {np.round(w, 3)} sum {w.sum():.3f}")
w, _ = nnls(L[val], df.P_fus.values[val]); MB_FIR = w / w.sum()
def mbfilt(Xsrc):
    out = np.zeros(Xsrc.shape)
    for wk, kq in zip(MB_FIR, LAGS):
        if wk > 0: out += wk * shift(Xsrc.values, kq)
    return pd.DataFrame(out, columns=Xsrc.columns, index=Xsrc.index)
Xm = mbfilt(Xc)
vm = np.isfinite(Xm.values).all(1) & (df.fsm != 1).values & df.P_fus.notna().values & (segi >= K)
dm = df[vm]; Xm = Xm[vm]; ym = dm.P_fus.values
print("fused 10 Hz rows:", len(ym), " BMS rows:", len(y))

def eval_bms(C, cols):
    p = np.zeros(len(y))
    for b in C.columns:
        te = (meta.boot == b).values; p[te] = Xf.loc[te, cols].values @ C[b].reindex(cols).values
    s = g.summarize(meta, p, y); e = pd.Series(p - y).groupby(meta.segid.values).mean()
    return dict(bms_seg_mae=s["seg_mae"], bms_seg_med=e.abs().median(), bms_seg_p90=s["seg_p90"], bms_seg_max=s["seg_max"], bms_boot_min=s["boot_min"], bms_boot_max=s["boot_max"]), p
def eval_fus(C, cols):
    p = np.zeros(len(ym))
    for b in C.columns:
        te = (dm.boot == b).values; p[te] = Xm.loc[te, cols].values @ C[b].reindex(cols).values
    e = p - ym; st = ~dm.action.isin(g.LOCO).values
    win = (dm.groupby("segid").cumcount() // 20).values
    w2 = pd.Series(e).groupby([dm.segid.values, win]).mean(); w2st = pd.Series(e[st]).groupby([dm.segid.values[st], win[st]]).mean()
    w2lo = pd.Series(e[~st]).groupby([dm.segid.values[~st], win[~st]]).mean()
    pk = pd.DataFrame({"m": ym[~st], "p": p[~st]}).groupby(dm.segid.values[~st]).max()
    return dict(f_rmse_static=np.sqrt((e[st] ** 2).mean()), f_rmse_loco=np.sqrt((e[~st] ** 2).mean()), f_2s_mae_static=w2st.abs().mean(), f_2s_p99_static=w2st.abs().quantile(.99),
                f_2s_mae_loco=w2lo.abs().mean(), f_2s_p99_loco=w2lo.abs().quantile(.99), loco_peak_bias=(pk.p - pk.m).mean()), p
def lobo_fus(cols, **kw):
    C = {}
    for b in sorted(dm.boot.unique()):
        tr = (dm.boot != b).values; C[b] = g.fit_bounded(Xm.loc[tr, cols], ym[tr], **kw)
    return pd.DataFrame(C)
def lobo_both(cols, wf, **kw):
    """stack BMS rows and fused rows; fused rows weighted by wf"""
    C = {}
    for b in sorted(dm.boot.unique()):
        tr1 = (meta.boot != b).values; tr2 = (dm.boot != b).values
        XX = pd.concat([Xf.loc[tr1, cols], Xm.loc[tr2, cols] * wf], ignore_index=True); yy = np.r_[y[tr1], ym[tr2] * wf]
        C[b] = g.fit_bounded(XX, yy, **kw)
    return pd.DataFrame(C)

rows = []
C_f = lobo_fus(BASE, **KW)
nb, nf = len(y), len(ym)
C_both = lobo_both(BASE, np.sqrt(nb / nf), **KW)          # each sensor carries the same total weight
for label, C in [("fit on BMS (2 Hz)", C_bms), ("fit on fused mainboard (10 Hz)", C_f), ("fit on both, equal total weight", C_both)]:
    a, _ = eval_bms(C, BASE); b2, _ = eval_fus(C, BASE); rows.append(dict(fit=label, **a, **b2))
print("\n== (b) which target to fit on; every number is for the held-out boot (W) =="); print(pd.DataFrame(rows).set_index("fit").round(2).T.to_string())
pd.DataFrame(rows).round(3).to_csv(os.path.join(A, "v2_dual_sensor_comparison.csv"), index=False, encoding="utf-8-sig")

# bootstrap by segment: coefficient spread for the three fits
rng = np.random.default_rng(0)
segs = np.array(sorted(set(meta.segid) | set(dm.segid)))
ib = {s: np.where(meta.segid.values == s)[0] for s in segs}; im = {s: np.where(dm.segid.values == s)[0] for s in segs}
Xb_, Xm_ = Xf[BASE].reset_index(drop=True), Xm[BASE].reset_index(drop=True)
wf = np.sqrt(nb / nf); bs = {"BMS": [], "fused": [], "both": []}
for r in range(200):
    pick = rng.choice(segs, len(segs))
    i1 = np.concatenate([ib[s] for s in pick]); i2 = np.concatenate([im[s] for s in pick])
    bs["BMS"].append(g.fit_bounded(Xb_.iloc[i1], y[i1], **KW))
    bs["fused"].append(g.fit_bounded(Xm_.iloc[i2], ym[i2], **KW))
    bs["both"].append(g.fit_bounded(pd.concat([Xb_.iloc[i1], Xm_.iloc[i2] * wf], ignore_index=True), np.r_[y[i1], ym[i2] * wf], **KW))
tab = {}
c_fus = g.fit_bounded(Xm[BASE], ym, **KW); c_both = g.fit_bounded(pd.concat([Xb_, Xm_ * wf], ignore_index=True), np.r_[y, ym * wf], **KW)
for nm, c0 in [("BMS", c_bms), ("fused", c_fus), ("both", c_both)]:
    B = pd.DataFrame(bs[nm]); tab[(nm, "coef")] = c0; tab[(nm, "lo")] = B.quantile(.025); tab[(nm, "hi")] = B.quantile(.975)
ctab = pd.DataFrame(tab); print("\ncoefficients and 95% bootstrap intervals (200 resamples of segments):"); print(ctab.round(4).to_string())
ctab.round(5).to_csv(os.path.join(A, "v2_dual_sensor_coefs.csv"), encoding="utf-8-sig")

# ------------------------------------------------------------------ (c) thermal creep
print("\n== (c) thermal creep term; held-out errors (W) ==")
rows = []
hold_m = dm.arm_holding.fillna(False).astype(bool).values & (dm.action == "reject").values
def hold_slope(p):
    """mean slope of (measured - model) against time inside the reject holds, W per 10 s"""
    out = []
    for s, x in pd.DataFrame({"t": dm.t.values[hold_m], "r": (ym - p)[hold_m]}).groupby(dm.segid.values[hold_m]):
        if len(x) > 40: out.append(np.polyfit(x.t - x.t.iloc[0], x.r, 1)[0] * 10)
    return float(np.mean(out)), len(out)
for label, extra in [("no creep term", []), ("creep arm, no leak", ["creep_arm_inf"]), ("creep arm, 60 s leak", ["creep_arm_60"]), ("creep arm, 20 s leak", ["creep_arm_20"]),
                     ("creep arm+leg+waist, no leak", ["creep_arm_inf", "creep_leg_inf", "creep_waist_inf"]), ("creep arm+leg+waist, 60 s leak", ["creep_arm_60", "creep_leg_60", "creep_waist_60"])]:
    cols = BASE + extra
    for tgt in ["BMS", "fused"]:
        if tgt == "BMS": _, C = g.lobo_cv(meta, Xf[cols], y, meta.boot.values, **KW); c_all = g.fit_bounded(Xf[cols], y, **KW)
        else: C = lobo_fus(cols, **KW); c_all = g.fit_bounded(Xm[cols], ym, **KW)
        a, _ = eval_bms(C, cols); b2, pf = eval_fus(C, cols); sl, ns = hold_slope(pf)
        rows.append(dict(model=label, target=tgt, bms_seg_mae=a["bms_seg_mae"], bms_seg_max=a["bms_seg_max"], f_rmse_static=b2["f_rmse_static"], f_2s_mae_static=b2["f_2s_mae_static"],
                         f_rmse_loco=b2["f_rmse_loco"], reject_hold_resid_slope_W_per_10s=sl, n_holds=ns, **{e: c_all[e] for e in extra}))
R = pd.DataFrame(rows); print(R.round(4).to_string(index=False))
R.round(5).to_csv(os.path.join(A, "v2_thermal_creep.csv"), index=False, encoding="utf-8-sig")
json.dump({"mb_fir_lags": LAGS, "mb_fir": [float(v) for v in MB_FIR]}, open(os.path.join(A, "v2_mb_fir.json"), "w"))
