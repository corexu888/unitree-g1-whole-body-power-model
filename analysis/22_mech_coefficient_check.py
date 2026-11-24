"""Energy-conservation check of the power scale without an external meter.
tau*dq is mechanical power in SI watts, computed from quantities that do not involve the battery current sensor.
If the coefficient of positive mechanical power is left free, a value near 1 means the BMS power scale is consistent with it."""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1power as g
ROOT = sys.argv[1]; A = os.path.join(ROOT, "analysis")
pd.set_option("display.width", 220)
df = g.load(ROOT)
X16T = g.features(df, g.V2_SPEC); X16T.insert(0, "const", 1.0); X = g.tie_types(X16T)
X["mech_net"] = X.mech_pos + X.mech_neg
segv = df.segid.values; segi = df.groupby("segid", sort=False).cumcount().values
K = len(g.BMS_FIR) - 1

# ---- target 1: BMS
meta_all, Xf_all = g.bms_align(df, X); k = (meta_all.fsm != 1).values
meta, Xf = meta_all[k], Xf_all[k].reset_index(drop=True); y = meta.P_bms.values
# ---- target 2: main-board power referenced to the BMS level per session (offset from the two sensors only)
df["P_mb_f"] = df.groupby("segid").P_mb.transform(lambda s: np.convolve(s.values, g.BMS_FIR, mode="full")[:len(s)])
df.loc[segi < K, "P_mb_f"] = np.nan
fresh = df.groupby("segid").cur_raw.transform(lambda s: np.r_[False, np.diff(s.values) != 0]).values.astype(bool)
ok = fresh & df.P_mb_f.notna().values
off = (df.P_bms - df.P_mb_f)[ok].groupby(df.boot[ok]).mean()
df["P_fus"] = df.P_mb + df.boot.map(off)
fir = json.load(open(os.path.join(A, "v2_mb_fir.json")))
Xm = np.zeros(X.shape)
for wk, lag in zip(fir["mb_fir"], fir["mb_fir_lags"]):
    if wk > 0: Xm += wk * pd.DataFrame(X.values).groupby(segv).shift(lag).values
vm = np.isfinite(Xm).all(1) & (df.fsm != 1).values & df.P_fus.notna().values & (segi >= K)
dm = df[vm]; Xm = pd.DataFrame(Xm[vm], columns=X.columns); ym = dm.P_fus.values

CU = [c for c in X.columns if c.startswith("cu_")]
VAR = {
 "v2: positive mech fixed at 1, braking in [0,1]": (["const"] + CU + ["mech_pos", "mech_neg"], dict(fixed={"mech_pos": 1.0}, upper={"mech_neg": 1.0})),
 "positive mech free, braking in [0,1]": (["const"] + CU + ["mech_pos", "mech_neg"], dict(upper={"mech_neg": 1.0})),
 "positive mech free, braking free": (["const"] + CU + ["mech_pos", "mech_neg"], dict()),
 "one free coefficient on net mechanical power": (["const"] + CU + ["mech_net"], dict()),
}
rng = np.random.default_rng(2)
def boot_ci(Xs, ys, seg_ids, cols, kw, n=300):
    segs = np.array(sorted(set(seg_ids))); rows = {s: np.where(seg_ids == s)[0] for s in segs}; out = []
    for r in range(n):
        idx = np.concatenate([rows[s] for s in rng.choice(segs, len(segs))])
        out.append(g.fit_bounded(Xs[cols].iloc[idx], ys[idx], **kw))
    return pd.DataFrame(out)
res = []
for tgt, Xs, ys, md in [("BMS", Xf, y, meta), ("main board + session offset", Xm, ym, dm)]:
    for name, (cols, kw) in VAR.items():
        c = g.fit_bounded(Xs[cols], ys, **kw); B = boot_ci(Xs, ys, md.segid.values, cols, kw, 300 if tgt == "BMS" else 150)
        row = dict(target=tgt, model=name, P0=c["const"])
        for nm in ["mech_pos", "mech_neg", "mech_net"]:
            if nm in cols: row[nm] = c[nm]; row[nm + "_lo"] = B[nm].quantile(.025); row[nm + "_hi"] = B[nm].quantile(.975)
        if tgt == "BMS":
            p, _ = g.lobo_cv(md, Xs[cols], ys, md.boot.values, **kw); e = pd.Series(p - ys).groupby(md.segid.values).mean()
            row["heldout_session_seg_mae"] = e.abs().mean(); row["heldout_session_seg_max"] = e.abs().max()
        res.append(row)
R = pd.DataFrame(res); print(R.round(3).to_string(index=False))
R.round(4).to_csv(os.path.join(A, "mech_coefficient_check.csv"), index=False, encoding="utf-8-sig")
# how much mechanical power is there to identify the coefficient from?
loco = meta.action.isin(g.LOCO).values
print("mean positive mechanical power (BMS rows): locomotion %.1f W, other %.1f W; braking: locomotion %.1f W, other %.1f W" % (
    Xf.mech_pos[loco].mean(), Xf.mech_pos[~loco].mean(), Xf.mech_neg[loco].mean(), Xf.mech_neg[~loco].mean()))
