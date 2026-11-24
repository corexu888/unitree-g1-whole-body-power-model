"""Analyses added for the revised manuscript (draft v2), in answer to review comments. Existing data only.
1 per-session results and session-weighted error          2 session-clustered bootstrap of the coefficients
3 prediction interval with variance parts estimated outside the test fold
4 gesture decomposition with intervals, two-arm subset, regressor correlations
5 temperature factor: dependence on where the record is cut; cool and warm starts
6 standing power before and after a motion in the same segment (paired)
7 how the segment subsets (76 / 74 / 75) are selected; supplementary segment table"""
import sys, os, json
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1power as g
ROOT = sys.argv[1]; A = os.path.join(ROOT, "analysis"); MS = os.path.join(ROOT, "manuscript")
pd.set_option("display.width", 240, "display.max_columns", 50, "display.max_rows", 200)
df = g.load(ROOT)
M = json.load(open(os.path.join(ROOT, "g1_power_model_v2.json"), encoding="utf-8"))
KW = dict(fixed={"mech_pos": 1.0}, upper={"mech_neg": 1.0})
tau, dq, T = g.arrays(df)
TYPE = [n.replace("left_", "").replace("right_", "") for n in g.JN]
GRP_OF = {t: gn for gn, types in M["groups"].items() for t in types}
bj = np.array([M["b"][t] for t in TYPE])
segi = df.groupby("segid", sort=False).cumcount().values
X0 = g.features(df, dict(cu="type", mech="split")); X0.insert(0, "const", 1.0); X0 = g.tie_types(X0)        # no temperature factor
X = g.features(df, g.V2_SPEC); X.insert(0, "const", 1.0); X = g.tie_types(X)                               # v2
meta_all, Xcat = g.bms_align(df, pd.concat([X, X0.add_suffix("@0")], axis=1)); k = (meta_all.fsm != 1).values
meta = meta_all[k]; Xf = Xcat[list(X.columns)][k]; Xf0 = Xcat[[c + "@0" for c in X.columns]][k].rename(columns=lambda c: c[:-2])
y = meta.P_bms.values; segid = meta.segid.values; boots = meta.boot.values
seg_info = meta.groupby("segid").agg(action=("action", "first"), boot=("boot", "first"), date=("date", "first"), robot=("robot", "first"))
ybar = pd.Series(y).groupby(segid).mean()
rng = np.random.default_rng(7)

# ================================================================== 1. per-session results
p, C = g.lobo_cv(meta, Xf, y, boots, **KW); p0, _ = g.lobo_cv(meta, Xf0, y, boots, **KW)
e = pd.Series(p - y).groupby(segid).mean()
ses = pd.DataFrame({"e": e}).join(seg_info).groupby("boot").agg(robot=("robot", "first"), date=("date", "first"), segments=("e", "size"), mae=("e", lambda s: s.abs().mean()),
                                                               bias=("e", "mean"), largest=("e", lambda s: s.abs().max()))
print("== 1. held-out session: one row per fold (W) =="); print(ses.round(2).to_string())
print("MAE weighted by segment %.2f W; mean of the nine session MAEs %.2f W; median of session MAEs %.2f W" % (e.abs().mean(), ses.mae.mean(), ses.mae.median()))
ses.round(3).to_csv(os.path.join(A, "rev_per_session.csv"), encoding="utf-8-sig")

# ================================================================== 2. session-clustered bootstrap
sess = np.array(sorted(set(boots))); rows_b = {b: np.where(boots == b)[0] for b in sess}; Xr = Xf.reset_index(drop=True)
BS = []
for r in range(500):
    idx = np.concatenate([rows_b[b] for b in rng.choice(sess, len(sess))])
    BS.append(g.fit_bounded(Xr.iloc[idx], y[idx], **KW))
BS = pd.DataFrame(BS)
ct = pd.read_csv(os.path.join(A, "v2_coef_table.csv"), index_col=0)
cmp_ = pd.DataFrame({"coef": ct.coef, "seg_lo": ct.lo, "seg_hi": ct.hi, "session_lo": BS.quantile(.025), "session_hi": BS.quantile(.975)})
cmp_["width_ratio"] = (cmp_.session_hi - cmp_.session_lo) / (cmp_.seg_hi - cmp_.seg_lo).replace(0, np.nan)
print("\n== 2. 95% intervals: resampling segments (300) vs resampling whole sessions (500) =="); print(cmp_.round(4).to_string())
cmp_.round(5).to_csv(os.path.join(A, "rev_coef_session_bootstrap.csv"), encoding="utf-8-sig")

# ================================================================== 3. nested prediction interval
rows_of = {s: np.where(segid == s)[0] for s in sorted(set(segid))}
out = []
for b in sess:
    tr = boots != b; inner = []
    for b2 in sess[sess != b]:
        tr2 = tr & (boots != b2); te2 = boots == b2
        c2 = g.fit_bounded(Xf[tr2], y[tr2], **KW)
        inner.append(pd.Series(Xf.values[te2] @ c2.values - y[te2]).groupby(segid[te2]).mean())
    s_ses = np.std([x.mean() for x in inner], ddof=1); s_seg = np.concatenate([(x - x.mean()).values for x in inner]).std(ddof=1)
    c = g.fit_bounded(Xf[tr], y[tr], **KW).values
    tr_segs = np.array(sorted(set(segid[tr]))); Bf = []
    for r in range(100):
        idx = np.concatenate([rows_of[s] for s in rng.choice(tr_segs, len(tr_segs))]); Bf.append(g.fit_bounded(Xr.iloc[idx], y[idx], **KW).values)
    Bf = np.array(Bf)
    for s in sorted(set(segid[~tr])):
        xb = Xr.iloc[rows_of[s]].mean().values
        out.append(dict(boot=b, segid=s, err=xb @ c - ybar[s], sd_par=(Bf @ xb).std(), sd_ses=s_ses, sd_seg=s_seg))
N = pd.DataFrame(out); N["sd"] = np.sqrt(N.sd_par ** 2 + N.sd_ses ** 2 + N.sd_seg ** 2); N["in95"] = N.err.abs() <= 1.96 * N.sd; N["in68"] = N.err.abs() <= N.sd
print("\n== 3. prediction interval, all variance parts estimated without the test session ==")
print("coverage of +-1.96 sd: %d of %d segments (%.0f%%); of +-1 sd: %.0f%%; mean sd %.2f W (session part %.2f-%.2f, segment part %.2f-%.2f)" % (
    N.in95.sum(), len(N), 100 * N.in95.mean(), 100 * N.in68.mean(), N.sd.mean(), N.sd_ses.min(), N.sd_ses.max(), N.sd_seg.min(), N.sd_seg.max()))
print(N.groupby("boot").agg(n=("err", "size"), covered=("in95", "sum"), sd=("sd", "mean"), sd_session_part=("sd_ses", "first")).round(2).T.to_string())
N.round(3).to_csv(os.path.join(A, "rev_nested_interval.csv"), index=False, encoding="utf-8-sig")

# ================================================================== 4. gesture decomposition
GEST = ["hands up", "heart", "high five", "hug", "reject", "shake hand", "xray"]
def phases(gg):
    """pre-motion standing and static hold of a gesture segment (same rules as 14_paper_model_comparison.py); returns (pre, hold, reason)"""
    if len(gg) < 150: return None, None, "fewer than 150 frames"
    if (gg.fsm == 1).any(): return None, None, "contains damping frames"
    dqa = gg[[f"dq{j}" for j in g.ARM]].abs().sum(1).values; mv = np.where(dqa > 1.0)[0]
    hold = gg.arm_holding.fillna(False).astype(bool).values
    if len(mv) == 0 or hold.sum() < 40: return None, None, "no arm motion or hold flag shorter than 40 frames"
    ms = mv[0]; h0 = np.where(hold)[0][0]; h1 = np.where(hold)[0][-1]; idx = np.arange(len(gg))
    pre = (idx >= 8) & (idx < ms - 5); hd = hold & (dqa < 0.6) & (idx >= h0 + 20) & (idx <= h1 - 3)
    if pre.sum() < 15: return None, None, "pre-motion standing shorter than 15 frames"
    if hd.sum() < 20: return None, None, "static hold shorter than 20 frames"
    return pre, hd, ""
parts, reasons = [], {}
for sid, gg in df[df.action.isin(GEST)].groupby("segid", sort=False):
    gg = gg.sort_values("t").reset_index(drop=True); pre, hd, why = phases(gg)
    if pre is None: reasons[sid] = why; continue
    Tg = gg[[f"T{j}" for j in range(29)]].values.astype(float)
    t2 = gg[[f"tau{j}" for j in range(29)]].values.astype(float) ** 2 * (1.0 + g.ALPHA_CU * (Tg - Tg[:15].mean(0)))      # copper regressor of v2, with the temperature factor
    parts.append(dict(segid=sid, action=gg.action.iloc[0], boot=gg.boot.iloc[0], meas=gg.P_bms.values[hd].mean() - gg.P_bms.values[pre].mean(), d=t2[hd].mean(0) - t2[pre].mean(0)))
P = pd.DataFrame(parts); D = np.array(P.d.tolist())
S = np.array(M["uncertainty"]["bootstrap_samples"]); gnames = list(M["groups"].keys())
BJ = np.array([[S[r, 1 + gnames.index(GRP_OF[t])] for t in TYPE] for r in range(len(S))])            # (300, 29) per-joint coefficients of each bootstrap set
def contrib(b):
    c = D * b; return c[:, g.ARM].sum(1), c[:, g.WAIST].sum(1), c[:, g.LEG].sum(1)
P["arms"], P["waist"], P["legs"] = contrib(bj); P["left"] = (D * bj)[:, 15:22].sum(1); P["right"] = (D * bj)[:, 22:29].sum(1)
bs_parts = np.array([np.column_stack(contrib(BJ[r])) for r in range(len(BJ))])                        # (300, 74, 3)
tab = []
for a, x in P.groupby("action"):
    ii = x.index.values; q = np.quantile(bs_parts[:, ii, :].mean(1), [.025, .975], axis=0)
    weak = min(x.left.mean(), x.right.mean()) / (x.left.mean() + x.right.mean())
    tab.append(dict(action=a, kind="two-arm" if weak >= 0.25 else "one-arm", n=len(x), meas=x.meas.mean(), arms=x.arms.mean(), arms_lo=q[0, 0], arms_hi=q[1, 0], waist=x.waist.mean(), waist_lo=q[0, 1], waist_hi=q[1, 1],
                    legs=x.legs.mean(), legs_lo=q[0, 2], legs_hi=q[1, 2], wl_share=(x.waist + x.legs).sum() / x.meas.sum()))
TB = pd.DataFrame(tab).set_index("action").sort_values("meas")
print("\n== 4. net power during the hold by gesture; intervals from the 300 bootstrap coefficient sets (W) =="); print(TB.round(2).to_string())
TB.round(3).to_csv(os.path.join(A, "rev_gesture_decomposition.csv"), encoding="utf-8-sig")
two = TB.index[TB.kind == "two-arm"].tolist()
def share_ci(mask, n=3000):
    ii = np.where(mask)[0]; vals = []
    for _ in range(n):
        jj = rng.choice(ii, len(ii)); r = rng.integers(len(BJ))
        vals.append(bs_parts[r, jj, 1:].sum() / P.meas.values[jj].sum())
    return np.quantile(vals, [.025, .975])
for label, mask in [("all gestures", np.ones(len(P), bool)), ("two-arm gestures (%s)" % ", ".join(two), P.action.isin(two).values),
                    ("two-arm gestures without reject", P.action.isin([a for a in two if a != "reject"]).values), ("one-arm gestures", (~P.action.isin(two)).values)]:
    x = P[mask]; lo, hi = share_ci(mask)
    print("%-62s n=%2d: waist+legs %.1f W / measured %.1f W = %.1f%% (95%% %.1f-%.1f%%, segments and coefficients resampled); segments above 50%%: %d" % (
        label, len(x), (x.waist + x.legs).mean(), x.meas.mean(), 100 * (x.waist + x.legs).sum() / x.meas.sum(), 100 * lo, 100 * hi, int(((x.waist + x.legs) / x.meas > 0.5).sum())))
cu = [c for c in Xf.columns if c.startswith("cu_")]; R = Xf[cu].corr(); vif = pd.Series(np.diag(np.linalg.inv(R.values)), index=cu)
print("correlation between copper regressors (fitting rows), pairs with |r| > 0.6:")
for i, a in enumerate(cu):
    for b_ in cu[i + 1:]:
        if abs(R.loc[a, b_]) > 0.6: print("   %s ~ %s: r = %.2f" % (a, b_, R.loc[a, b_]))
print("r(N7520_14, N7520_22) = %.2f; variance inflation factors:" % R.loc["cu_N7520_14", "cu_N7520_22"], vif.round(1).to_dict())

# ================================================================== 5. temperature factor
print("\n== 5. temperature factor ==")
hold_fr = df.arm_holding.fillna(False).astype(bool).values & (df.fsm != 1).values
T0 = pd.DataFrame(np.where((segi < 15)[:, None], T, np.nan)).groupby(df.segid.values).transform("mean").values
rec = []
for sid, idx in df[hold_fr].groupby("segid", sort=False).indices.items():
    ii = np.where(hold_fr)[0][idx]; h0 = ii[0]
    Th0 = T[h0:h0 + 15].mean(0)
    shift = (bj * tau[ii] ** 2 * g.ALPHA_CU * (Th0 - T0[h0])).sum(1).mean()          # W by which the prediction drops if the reference is taken at hold start
    rec.append(dict(segid=sid, action=df.action.values[h0], cut_shift_W=shift, Tstart_arm=T0[h0][g.ARM].mean(), rise_before_hold=(Th0 - T0[h0])[g.ARM].max(), hold_s=df.t.values[ii[-1]] - df.t.values[ii[0]]))
H = pd.DataFrame(rec).set_index("segid")
hm = meta.arm_holding.fillna(False).astype(bool).values
def slopes(pred):
    return pd.Series({s: np.polyfit(x.t - x.t.iloc[0], x.r, 1)[0] * 10 for s, x in pd.DataFrame({"t": meta.t.values[hm], "r": (y - pred)[hm]}).groupby(segid[hm]) if len(x) >= 12})
H["drift_no_factor"] = slopes(p0); H["drift_v2"] = slopes(p); H["hold_bms_samples"] = pd.Series(segid[hm]).value_counts()
H["set"] = np.where(H.action == "reject", "reject", "other gestures")
print("if the record were cut at the start of the hold (reference taken there), the predicted hold power would be lower by:")
print(H.groupby("set").cut_shift_W.agg(["size", "mean", "max"]).round(2).to_string())
print("largest temperature rise of an arm motor between segment start and hold start: mean %.1f K (reject %.1f K)" % (H.rise_before_hold.mean(), H[H.set == "reject"].rise_before_hold.mean()))
Hd = H.dropna(subset=["drift_no_factor"]).copy(); Hd["start"] = np.where(Hd.Tstart_arm <= Hd.groupby("set").Tstart_arm.transform("median"), "cooler half", "warmer half")
print("drift inside holds (W per 10 s) by arm temperature at segment start:")
print(Hd.groupby(["set", "start"]).agg(n=("drift_v2", "size"), T_start=("Tstart_arm", "mean"), no_factor=("drift_no_factor", "mean"), v2=("drift_v2", "mean")).round(2).to_string())
print("holds in the drift analysis:", len(Hd), "| gesture segments with a hold flag:", len(H), "| not used (fewer than 12 BMS samples in the hold):", H.index[H.drift_no_factor.isna()].tolist())
H.round(3).to_csv(os.path.join(A, "rev_temperature_holds.csv"), encoding="utf-8-sig")

# ================================================================== 6. paired stance comparison inside a segment
# First pass: windows defined from whole-body motion, which leaves few pairs. The manuscript uses the version in
# 26_review2_checks.py (part A), where the windows are defined from arm motion; that script also fills the supplement columns.
print("\n== 6. standing before and after the motion of the same segment (robot B, arms down, quiet); first pass, see 26_review2_checks.py ==")
coef = np.array([M["P0"]] + [M["b_group"][gn] for gn in gnames] + [1.0, M["eta_regen"]])
df["model"] = X[["const"] + [f"cu_{gn}" for gn in gnames] + ["mech_pos", "mech_neg"]].values @ coef
df["speed"] = np.abs(dq).sum(1); df["armcu"] = (bj[g.ARM] * tau[:, g.ARM] ** 2).sum(1)
df["qa"] = np.degrees(df[["q4", "q10"]].mean(1)); df["qk"] = np.degrees(df[["q3", "q9"]].mean(1)); df["Tank"] = T[:, [4, 5, 10, 11]].mean(1); df["Tarm"] = T[:, g.ARM].mean(1)
rec = []
for sid, gg in df[(df.robot == "B") & ~df.action.isin(g.LOCO + ["squat"])].groupby("segid", sort=False):
    gg = gg.sort_values("t").reset_index(drop=True); idx = np.arange(len(gg))
    if len(gg) < 150 or (gg.fsm == 1).any() or gg.qa.isna().all(): continue
    mv = np.where(gg.speed.values >= 0.5)[0]; quiet = (gg.speed.values < 0.5) & (gg.armcu.values < 3.0)
    if len(mv) < 10: continue                                   # standing-only segments have no before/after
    pre = quiet & (idx >= 8) & (idx < mv[0] - 5); post = quiet & (idx >= mv[-1] + 20)          # 2 s after the last motion: BMS response has settled
    if pre.sum() < 15 or post.sum() < 15: continue
    a, b_ = gg[pre], gg[post]
    rec.append(dict(segid=sid, action=gg.action.iloc[0], boot=gg.boot.iloc[0], P_pre=a.P_bms.mean(), P_post=b_.P_bms.mean(), qa_pre=a.qa.mean(), qa_post=b_.qa.mean(), qk_pre=a.qk.mean(), qk_post=b_.qk.mean(),
                    m_pre=a.model.mean(), m_post=b_.model.mean(), Tank_d=b_.Tank.mean() - a.Tank.mean(), Tarm_d=b_.Tarm.mean() - a.Tarm.mean(), gap_s=b_.t.mean() - a.t.mean()))
Q = pd.DataFrame(rec); thr = np.degrees(-0.2)
Q["trans"] = np.where(Q.qa_pre < thr, "deep", "shallow") + " -> " + np.where(Q.qa_post < thr, "deep", "shallow")
Q["dP"] = Q.P_post - Q.P_pre; Q["dqa"] = Q.qa_post - Q.qa_pre; Q["dqk"] = Q.qk_post - Q.qk_pre; Q["dM"] = Q.m_post - Q.m_pre
def ci(v):
    v = np.asarray(v, float); h = stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 2 else np.nan; return v.mean() - h, v.mean() + h
rows = []
for tname, x in Q.groupby("trans"):
    lo, hi = ci(x.dP)
    rows.append(dict(transition=tname, n=len(x), sessions=x.boot.nunique(), P_before=x.P_pre.mean(), P_after=x.P_post.mean(), dP=x.dP.mean(), dP_lo=lo, dP_hi=hi, d_model=x.dM.mean(), d_ankle_deg=x.dqa.mean(), d_knee_deg=x.dqk.mean(),
                     d_T_ankle=x.Tank_d.mean(), d_T_arm=x.Tarm_d.mean(), gap_s=x.gap_s.mean()))
QT = pd.DataFrame(rows); print(QT.round(2).to_string(index=False))
a = Q[Q.trans == "deep -> shallow"].dP; b_ = Q[Q.trans == "deep -> deep"].dP
if len(a) > 2 and len(b_) > 2:
    d = a.mean() - b_.mean(); se = np.sqrt(a.var(ddof=1) / len(a) + b_.var(ddof=1) / len(b_))
    print("deep->shallow minus deep->deep: %.1f W (95%% %.1f to %.1f)" % (d, d - 1.96 * se, d + 1.96 * se))
QT.round(3).to_csv(os.path.join(A, "rev_stance_paired_firstpass.csv"), index=False, encoding="utf-8-sig")

# ================================================================== 7. subsets and supplementary segment table
print("\n== 7. segment subsets ==")
gseg = df[df.action.isin(GEST)].segid.unique()
print("gesture segments in the telemetry:", len(gseg), "| in the fit:", int(seg_info.action.isin(GEST).sum()), "| in the net-power set (Tables 5-6):", len(P), "| holds in the drift analysis:", len(Hd))
for s_, why in reasons.items(): print("   not in the net-power set:", s_, "|", why, "| frames", int((df.segid == s_).sum()))
sup = df.groupby("segid", sort=False).agg(date=("date", "first"), robot=("robot", "first"), session=("boot", "first"), action=("action", "first"), frames=("t", "size"), t_start=("t", "min"), t_end=("t", "max")).reset_index()
sup["session"] += 1; sup["duration_s"] = (sup.t_end - sup.t_start).round(1); sup = sup.sort_values("t_start").reset_index(drop=True); sup.insert(0, "no", np.arange(1, len(sup) + 1))
sup["bms_rows_in_fit"] = sup.segid.map(pd.Series(segid).value_counts()).fillna(0).astype(int)
sup["in_net_power_set"] = sup.segid.isin(P.segid).astype(int); sup["in_hold_drift_set"] = sup.segid.isin(Hd.index).astype(int)
sup["in_paired_stance_set"] = sup.segid.isin(Q.segid).astype(int); sup["stance_transition"] = sup.segid.map(Q.set_index("segid").trans).fillna("")
ls = np.abs(dq[:, g.LEG]).sum(1); w0, w1 = {}, {}
for sid, gg in df[df.action.isin(g.LOCO)].groupby("segid"):
    gg = gg.sort_values("t"); mvv = (pd.Series(ls[gg.index]).rolling(10, center=True, min_periods=1).mean() > 1.5).values
    if mvv.sum() >= 30:
        i0, i1 = np.where(mvv)[0][0], np.where(mvv)[0][-1]; w0[sid] = gg.t.values[i0 + 12] - gg.t.values[0]; w1[sid] = gg.t.values[i1 - 8] - gg.t.values[0]
sup["moving_window_start_s"] = sup.segid.map(w0).round(1); sup["moving_window_end_s"] = sup.segid.map(w1).round(1)
sup["segment"] = sup.segid.str.split("|").str[1]
sup = sup[["no", "date", "robot", "session", "segment", "action", "frames", "duration_s", "bms_rows_in_fit", "in_net_power_set", "in_hold_drift_set", "in_paired_stance_set", "stance_transition", "moving_window_start_s", "moving_window_end_s"]]
sup.to_csv(os.path.join(MS, "supplement_segments.csv"), index=False, encoding="utf-8-sig")
print("supplement table:", len(sup), "segments ->", os.path.join("manuscript", "supplement_segments.csv"))
print("BMS response weights (lags 0..8 frames):", [round(float(v), 4) for v in g.BMS_FIR])
