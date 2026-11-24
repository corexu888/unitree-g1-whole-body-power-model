"""Checks prompted by a review of the draft.
A. Does the held-out accuracy depend on the BMS response kernel (which was estimated from arm/standing data of all sessions)?
   Compare: identified kernel, boxcar, pure delay, no alignment, and fitting on segment means (no kernel at all).
B. Which gestures load one arm and which load two, and what share of the net hold power is in waist + legs for the two-arm subset?"""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1power as g
ROOT = sys.argv[1]; A = os.path.join(ROOT, "analysis")
pd.set_option("display.width", 230, "display.max_columns", 40)
df = g.load(ROOT)
KW = dict(fixed={"mech_pos": 1.0}, upper={"mech_neg": 1.0})
X16T = g.features(df, g.V2_SPEC); X16T.insert(0, "const", 1.0); X = g.tie_types(X16T)
FRAME_IN_SEG = df.groupby("segid", sort=False).cumcount()

# ------------------------------------------------------------------ A. kernel sensitivity
def box(a, b):
    f = np.zeros(b + 1); f[a:b + 1] = 1.0 / (b - a + 1); return f
def delta(l):
    f = np.zeros(l + 1); f[l] = 1.0; return f
KERNELS = {"identified response (9 taps)": g.BMS_FIR, "boxcar, lags 1-6": box(1, 6), "pure delay, 4 frames": delta(4), "no alignment (lag 0)": delta(0)}
rows = []
for name, fir in KERNELS.items():
    ma, Xa = g.bms_align(df, X, fir=fir); kq = (ma.fsm != 1).values; ma, Xa = ma[kq], Xa[kq]; y = ma.P_bms.values; sid = ma.segid.values
    gest = ~ma.action.isin(g.LOCO + ["stand still", "squat"]).values; loco = ma.action.isin(g.LOCO).values
    p, _ = g.lobo_cv(ma, Xa, y, ma.boot.values, **KW); e = p - y
    es = pd.Series(e).groupby(sid).mean()
    win = (FRAME_IN_SEG[ma.index] // 19).values                     # 2 s windows by frame index, as in 20_paper_figures.py
    w = pd.DataFrame({"e": e, "s": sid, "w": win, "loco": loco}).groupby(["s", "w"]).agg(e=("e", "mean"), n=("e", "size"), loco=("loco", "first")); w = w[w.n >= 2]
    c = g.fit_bounded(Xa[~gest], y[~gest], **KW); eg = pd.Series((Xa.values @ c.values - y)[gest]).groupby(sid[gest]).mean()
    rows.append(dict(scheme=name, rows=len(y), session_seg_mae=es.abs().mean(), session_seg_max=es.abs().max(), rmse_static=np.sqrt((e[~loco] ** 2).mean()), rmse_loco=np.sqrt((e[loco] ** 2).mean()),
                     win2s_static_mae=w.e[~w.loco].abs().mean(), win2s_loco_mae=w.e[w.loco].abs().mean(), nogesture_seg_mae=eg.abs().mean(), nogesture_seg_max=eg.abs().max()))
# fitting on segment means: one row per segment, no kernel
segi = df.groupby("segid", sort=False).cumcount().values
use = (segi >= 8) & (df.fsm != 1).values
S = X[use].groupby(df.segid[use].values).mean(); ys = df.P_bms[use].groupby(df.segid[use].values).mean().reindex(S.index).values
info = df[use].groupby("segid").agg(boot=("boot", "first"), action=("action", "first")).reindex(S.index)
p, _ = g.lobo_cv(info, S, ys, info.boot.values, **KW)
gest_s = ~info.action.isin(g.LOCO + ["stand still", "squat"]).values
c = g.fit_bounded(S[~gest_s], ys[~gest_s], **KW); eg = (S.values @ c.values - ys)[gest_s]
rows.append(dict(scheme="fit on segment means (no kernel)", rows=len(ys), session_seg_mae=np.abs(p - ys).mean(), session_seg_max=np.abs(p - ys).max(), nogesture_seg_mae=np.abs(eg).mean(), nogesture_seg_max=np.abs(eg).max()))
RA = pd.DataFrame(rows); print("== A. sensitivity to the sensor kernel (v2, W) =="); print(RA.round(2).to_string(index=False))
RA.round(3).to_csv(os.path.join(A, "review_kernel_sensitivity.csv"), index=False, encoding="utf-8-sig")

# ------------------------------------------------------------------ A2. kernel identified without the test data (nested)
from scipy.optimize import nnls
def est_kernel(mask, K=8):
    """BMS response from the two power sensors only (as 05_bms_response.py), using the frames in `mask`; per-session means removed."""
    rows = []
    for sid, idx in df[mask].groupby("segid", sort=False).indices.items():
        gg = df[mask].iloc[idx]
        pm = gg.P_mb.interpolate().bfill().values; pb = gg.P_bms.values; I = gg.cur_raw.values; t = gg.t.values
        chg = np.r_[False, np.diff(I) != 0]; okk = np.r_[True, np.diff(t) < 0.2]
        for i in np.where(chg)[0]:
            if i < K or not okk[i - K:i + 1].all() or not np.isfinite(pm[i - K:i + 1]).all(): continue
            rows.append((gg.boot.iloc[0], pb[i]) + tuple(pm[i - kk] for kk in range(K + 1)))
    R = pd.DataFrame(rows); Xk = R.iloc[:, 2:].values - R.groupby(0)[list(R.columns[2:])].transform("mean").values; yk = R[1].values - R.groupby(0)[1].transform("mean").values
    h, _ = nnls(Xk, yk); return h / h.sum(), len(R)
fsm_ok = (df.fsm != 1).values
nong = df.action.isin(g.LOCO + ["stand still", "squat"]).values
print("\n== A2. kernel identified without the test data ==")
print("kernel used in the draft      :", np.round(g.BMS_FIR, 3))
k_ng, n_ng = est_kernel(nong & fsm_ok); print("from non-gesture data only    :", np.round(k_ng, 3), " rows", n_ng, " mean delay %.2f s" % (0.103 * (np.arange(9) * k_ng).sum()))
ma, Xa = g.bms_align(df, X, fir=k_ng); kq = (ma.fsm != 1).values; ma, Xa = ma[kq], Xa[kq]; y = ma.P_bms.values; sid = ma.segid.values
gest = ~ma.action.isin(g.LOCO + ["stand still", "squat"]).values
c = g.fit_bounded(Xa[~gest], y[~gest], **KW); eg = pd.Series((Xa.values @ c.values - y)[gest]).groupby(sid[gest]).mean()
print("no gestures in kernel or fit -> gestures: segment MAE %.2f W, largest %.2f W (draft: 4.60 / 11.49)" % (eg.abs().mean(), eg.abs().max()))
errs = []; kern = {}
for b in sorted(df.boot.unique()):
    kb, nb_ = est_kernel((df.boot != b).values & fsm_ok); kern[b] = kb
    ma, Xa = g.bms_align(df, X, fir=kb); kq = (ma.fsm != 1).values; ma, Xa = ma[kq], Xa[kq]; y = ma.P_bms.values
    te = (ma.boot == b).values; c = g.fit_bounded(Xa[~te], y[~te], **KW)
    errs.append(pd.Series((Xa.values @ c.values - y)[te]).groupby(ma.segid.values[te]).mean())
E = pd.concat(errs)
print("held-out session, kernel and coefficients both from the other sessions: segment MAE %.2f W, median %.2f, largest %.2f W (draft: 1.65 / 1.11 / 5.79), n = %d" % (E.abs().mean(), E.abs().median(), E.abs().max(), len(E)))
print("spread of the nine fold kernels (max - min per tap):", np.round(np.ptp(np.array(list(kern.values())), axis=0), 3))

# ------------------------------------------------------------------ B. one-arm / two-arm gestures and the waist + leg share
M = json.load(open(os.path.join(ROOT, "g1_power_model_v2.json"), encoding="utf-8"))
bj = np.array([M["b"][n.replace("left_", "").replace("right_", "")] for n in g.JN])
GEST = ["hands up", "heart", "high five", "hug", "reject", "shake hand", "xray"]; LARM = list(range(15, 22)); RARM = list(range(22, 29))
parts = []
for sid, gg in df[df.action.isin(GEST)].groupby("segid", sort=False):
    gg = gg.sort_values("t").reset_index(drop=True)
    if len(gg) < 150 or (gg.fsm == 1).any(): continue
    dqa = gg[[f"dq{j}" for j in g.ARM]].abs().sum(1).values; mv = np.where(dqa > 1.0)[0]
    hold = gg.arm_holding.fillna(False).astype(bool).values
    if len(mv) == 0 or hold.sum() < 40: continue
    ms = mv[0]; h0 = np.where(hold)[0][0]; h1 = np.where(hold)[0][-1]; idx = np.arange(len(gg))
    pre = (idx >= 8) & (idx < ms - 5); hd = hold & (dqa < 0.6) & (idx >= h0 + 20) & (idx <= h1 - 3)
    if pre.sum() < 15 or hd.sum() < 20: continue
    t2 = gg[[f"tau{j}" for j in range(29)]].values.astype(float) ** 2
    d = bj * (t2[hd].mean(0) - t2[pre].mean(0))
    parts.append(dict(segid=sid, action=gg.action.iloc[0], boot=gg.boot.iloc[0], meas=gg.P_bms.values[hd].mean() - gg.P_bms.values[pre].mean(),
                      left=d[LARM].sum(), right=d[RARM].sum(), waist=d[g.WAIST].sum(), legs=d[g.LEG].sum()))
P = pd.DataFrame(parts); P["arms"] = P.left + P.right; P["wl"] = P.waist + P.legs
ga = P.groupby("action").agg(n=("meas", "size"), meas=("meas", "mean"), left=("left", "mean"), right=("right", "mean"), waist=("waist", "mean"), legs=("legs", "mean"))
ga["weaker_arm_share"] = ga[["left", "right"]].min(1) / (ga.left + ga.right)
ga["kind"] = np.where(ga.weaker_arm_share >= 0.25, "two-arm", "one-arm")
ga["waist_legs_share"] = (ga.waist + ga.legs) / ga.meas
print("\n== B. arm use by gesture (hold minus standing before the motion, W; v2 coefficients) =="); print(ga.round(2).to_string())
rng = np.random.default_rng(3)
def share(x): return x.wl.sum() / x.meas.sum()
def ci_seg(x, n=2000): return np.quantile([share(x.sample(len(x), replace=True, random_state=int(rng.integers(1e9)))) for _ in range(n)], [.025, .975])
def ci_boot(x, n=2000):
    bs = x.boot.unique(); out = []
    for _ in range(n):
        out.append(share(pd.concat([x[x.boot == b] for b in rng.choice(bs, len(bs))])))
    return np.quantile(out, [.025, .975])
two = ga.index[ga.kind == "two-arm"].tolist()
for label, x in [("all 74 segments", P), ("two-arm gestures: " + ", ".join(two), P[P.action.isin(two)]), ("two-arm without reject", P[P.action.isin([a for a in two if a != "reject"])]),
                 ("one-arm gestures", P[~P.action.isin(two)])]:
    if len(x) == 0: continue
    a, b = ci_seg(x); c2, d2 = ci_boot(x)
    print("%-60s n=%2d  waist+legs %.1f W of %.1f W measured = %.1f%%  (95%% by segment %.1f-%.1f%%, by session %.1f-%.1f%%; %d sessions) | segments with share > 50%%: %d" % (
        label, len(x), x.wl.mean(), x.meas.mean(), 100 * share(x), 100 * a, 100 * b, 100 * c2, 100 * d2, x.boot.nunique(), int(((x.wl / x.meas) > 0.5).sum())))
P.round(3).to_csv(os.path.join(A, "review_gesture_arm_use.csv"), index=False, encoding="utf-8-sig")
