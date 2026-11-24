"""Checks prompted by a second review. Existing data only.
A paired standing power before / after a gesture (refined windows)      B temperature reference taken in the middle of a hold
C what the BMS - main-board offset correlates with                      D held-out residual against voltage, state of charge, power level
E model variants: bus-level braking, pair-mean temperature for parallel joints, cross terms, |tau| terms, bus resistance
F gesture decomposition with the temperature factor included            G excitation statistics per actuator group
H does the absolute temperature level carry information?                I details asked for (Fig. 6 outlier, spread of the reject gesture)"""
import sys, os, json
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1power as g
ROOT = sys.argv[1]; A = os.path.join(ROOT, "analysis")
pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_rows", 200)
df = g.load(ROOT)
M = json.load(open(os.path.join(ROOT, "g1_power_model_v2.json"), encoding="utf-8"))
KW = dict(fixed={"mech_pos": 1.0}, upper={"mech_neg": 1.0})
tau, dq, T = g.arrays(df)
TYPE = [n.replace("left_", "").replace("right_", "") for n in g.JN]; gnames = list(M["groups"].keys())
GRP_OF = {t: gn for gn, types in M["groups"].items() for t in types}
bj = np.array([M["b"][t] for t in TYPE]); ALPHA = g.ALPHA_CU
segv = df.segid.values; segi = df.groupby("segid", sort=False).cumcount().values
T0 = pd.DataFrame(np.where((segi < 15)[:, None], T, np.nan)).groupby(segv).transform("mean").values
pm = tau * dq
def v2_features(Tm, T0m):
    t2 = tau ** 2 * (1.0 + ALPHA * (Tm - T0m)); cols = {"const": np.ones(len(df))}
    for gn, types in M["groups"].items(): cols[f"cu_{gn}"] = t2[:, [j for j, t in enumerate(TYPE) if t in types]].sum(1)
    cols["mech_pos"] = np.clip(pm, 0, None).sum(1); cols["mech_neg"] = np.clip(pm, None, 0).sum(1)
    return pd.DataFrame(cols, index=df.index)
X = v2_features(T, T0); BASE = list(X.columns)
coef_all = np.array([M["P0"]] + [M["b_group"][gn] for gn in gnames] + [1.0, M["eta_regen"]])
df["model"] = X.values @ coef_all
# extra regressors
Tp = T.copy()
for a_, b_ in [(4, 5), (10, 11), (13, 14)]: Tp[:, [a_, b_]] = T[:, [a_, b_]].mean(1, keepdims=True)
T0p = pd.DataFrame(np.where((segi < 15)[:, None], Tp, np.nan)).groupby(segv).transform("mean").values
XP = v2_features(Tp, T0p).add_suffix("@P")
E = pd.DataFrame(index=df.index)
net = pm.sum(1); E["netpos"] = np.clip(net, 0, None); E["netneg"] = np.clip(net, None, 0)
E["x_ankle"] = tau[:, 4] * tau[:, 5] + tau[:, 10] * tau[:, 11]; E["x_waist"] = tau[:, 13] * tau[:, 14]
E["abs_leg"] = np.abs(tau[:, g.LEG]).sum(1); E["abs_waist"] = np.abs(tau[:, g.WAIST]).sum(1); E["abs_arm"] = np.abs(tau[:, g.ARM]).sum(1)
E["i2"] = (df.model.values / df.volt_v.values) ** 2
cu_j = bj * tau ** 2
E["lvl_arm"] = (cu_j[:, g.ARM] * (T0[:, g.ARM] - 45.0)).sum(1); ANK = [4, 5, 10, 11]; OTH = [j for j in g.LEG + g.WAIST if j not in ANK]
E["lvl_ankle"] = (cu_j[:, ANK] * (T0[:, ANK] - 45.0)).sum(1); E["lvl_other"] = (cu_j[:, OTH] * (T0[:, OTH] - 45.0)).sum(1)
meta_all, Xcat = g.bms_align(df, pd.concat([X, XP, E], axis=1)); k = (meta_all.fsm != 1).values
meta = meta_all[k]; Xc = Xcat[k].reset_index(drop=True); y = meta.P_bms.values; segid = meta.segid.values; boots = meta.boot.values; acts = meta.action.values
seg_info = meta.groupby("segid").agg(action=("action", "first"), boot=("boot", "first"), robot=("robot", "first"))
gest_rows = ~meta.action.isin(g.LOCO + ["stand still", "squat"]).values
rng = np.random.default_rng(11)
GEST = ["hands up", "heart", "high five", "hug", "reject", "shake hand", "xray"]
df["speed"] = np.abs(dq).sum(1); df["armsp"] = np.abs(dq[:, g.ARM]).sum(1); df["armcu"] = cu_j[:, g.ARM].sum(1)
df["qa"] = np.degrees(df[["q4", "q10"]].mean(1)); df["qk"] = np.degrees(df[["q3", "q9"]].mean(1)); df["Tank"] = T[:, ANK].mean(1); df["ankW"] = (cu_j[:, ANK].sum(1)) / 4

# ================================================================== A. paired stance, refined windows
print("== A. standing before and after a gesture in the same segment (robot B; arms down and quiet in both windows) ==")
thr = np.degrees(-0.2); rec = []
for sid, gg in df[(df.robot == "B") & df.action.isin(GEST)].groupby("segid", sort=False):
    gg = gg.sort_values("t").reset_index(drop=True); idx = np.arange(len(gg))
    if len(gg) < 150 or (gg.fsm == 1).any(): continue
    am = np.where(gg.armsp.values > 1.0)[0]; hold = gg.arm_holding.fillna(False).astype(bool).values
    if len(am) < 10 or hold.sum() < 40: continue
    quiet = (gg.speed.values < 0.5) & (gg.armcu.values < 3.0)
    pre = quiet & (idx >= 8) & (idx < am[0] - 5); post = quiet & (idx >= am[-1] + 20)            # 2 s after the arms stop: BMS response has settled
    hd = hold & (gg.armsp.values < 0.6)
    r = dict(segid=sid, action=gg.action.iloc[0], boot=gg.boot.iloc[0], n_pre=int(pre.sum()), n_post=int(post.sum()), qa_pre=gg.qa[pre].mean(), qa_hold=gg.qa[hd].mean(),
             qa_post=gg.qa[post].mean() if post.sum() else np.nan, qk_pre=gg.qk[pre].mean(), qk_post=gg.qk[post].mean() if post.sum() else np.nan,
             P_pre=gg.P_bms[pre].mean(), P_post=gg.P_bms[post].mean() if post.sum() else np.nan, m_pre=gg.model[pre].mean(), m_post=gg.model[post].mean() if post.sum() else np.nan,
             Tank_pre=gg.Tank[pre].mean(), Tank_post=gg.Tank[post].mean() if post.sum() else np.nan)
    rec.append(r)
Q = pd.DataFrame(rec); st = lambda v: np.where(v < thr, "deep", "shallow")
print("gesture segments of robot B with a pre-motion standing window: %d; stance before the motion: %s; stance during the hold: %s" % (
    len(Q), pd.Series(st(Q.qa_pre)).value_counts().to_dict(), pd.crosstab(st(Q.qa_pre), st(Q.qa_hold)).to_dict()))
Qp = Q[(Q.n_pre >= 15) & (Q.n_post >= 15)].copy(); Qp["trans"] = pd.Series(st(Qp.qa_pre), index=Qp.index) + " -> " + pd.Series(st(Qp.qa_post), index=Qp.index)
Qp["dP"] = Qp.P_post - Qp.P_pre; Qp["dM"] = Qp.m_post - Qp.m_pre; Qp["dqa"] = Qp.qa_post - Qp.qa_pre; Qp["dqk"] = Qp.qk_post - Qp.qk_pre; Qp["dT"] = Qp.Tank_post - Qp.Tank_pre
def tci(v):
    v = np.asarray(v, float)
    if len(v) < 3: return np.nan, np.nan
    h = stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)); return v.mean() - h, v.mean() + h
rows = []
for tname, x in Qp.groupby("trans"):
    lo, hi = tci(x.dP)
    rows.append(dict(transition=tname, n=len(x), sessions=x.boot.nunique(), P_before=x.P_pre.mean(), P_after=x.P_post.mean(), dP=x.dP.mean(), dP_lo=lo, dP_hi=hi, d_model=x.dM.mean(),
                     ankle_before=x.qa_pre.mean(), ankle_after=x.qa_post.mean(), d_knee=x.dqk.mean(), d_T_ankle=x.dT.mean()))
QT = pd.DataFrame(rows); print(QT.round(2).to_string(index=False))
a = Qp[Qp.trans == "deep -> shallow"].dP; b_ = Qp[Qp.trans == "deep -> deep"].dP
if len(a) > 2 and len(b_) > 2:
    d = a.mean() - b_.mean(); se = np.sqrt(a.var(ddof=1) / len(a) + b_.var(ddof=1) / len(b_)); print("deep->shallow minus deep->deep: %.1f W (95%% %.1f to %.1f)" % (d, d - 1.96 * se, d + 1.96 * se))
QT.round(3).to_csv(os.path.join(A, "rev2_stance_paired.csv"), index=False, encoding="utf-8-sig"); Qp.round(3).to_csv(os.path.join(A, "rev2_stance_paired_segments.csv"), index=False, encoding="utf-8-sig")
supf = os.path.join(ROOT, "manuscript", "supplement_segments.csv")                     # written by 24_revision_analyses.py; fill the paired-stance columns here
if os.path.exists(supf):
    sup = pd.read_csv(supf); key = sup.date + "|" + sup.segment; tr = Qp.set_index("segid").trans
    sup["in_paired_stance_set"] = key.isin(tr.index).astype(int); sup["stance_transition"] = key.map(tr).fillna("")
    sup.to_csv(supf, index=False, encoding="utf-8-sig"); print("supplement table: %d segments in the paired stance set" % int(sup.in_paired_stance_set.sum()))

# ================================================================== B. reference taken in the middle of a hold
print("\n== B. temperature reference taken at mid-hold instead of segment start ==")
hold_fr = df.arm_holding.fillna(False).astype(bool).values & (df.fsm != 1).values; rec = []
for sid, idx in df[hold_fr].groupby("segid", sort=False).indices.items():
    ii = np.where(hold_fr)[0][idx]; mid = ii[len(ii) // 2]; late = ii[len(ii) // 2:]
    Tm = T[mid - 7:mid + 8].mean(0)
    rec.append(dict(action=df.action.values[ii[0]], shift=(cu_j[late] * ALPHA * (Tm - T0[mid])).sum(1).mean(), rise=float((Tm - T0[mid])[g.ARM].max()), power=df.P_bms.values[late].mean()))
B = pd.DataFrame(rec); B["set"] = np.where(B.action == "reject", "reject", "other gestures")
print(B.groupby("set").agg(n=("shift", "size"), shift_W=("shift", "mean"), shift_max=("shift", "max"), hottest_arm_rise_K=("rise", "mean"), power_second_half=("power", "mean")).round(2).to_string())

# ================================================================== C. sensor offset diagnostics
print("\n== C. BMS - main-board offset per segment against candidate causes ==")
K = len(g.BMS_FIR) - 1
df["P_mb_f"] = df.groupby("segid").P_mb.transform(lambda s: np.convolve(s.values, g.BMS_FIR, mode="full")[:len(s)]); df.loc[segi < K, "P_mb_f"] = np.nan
fresh = df.groupby("segid").cur_raw.transform(lambda s: np.r_[False, np.diff(s.values) != 0]).values.astype(bool) & df.P_mb_f.notna().values & (df.fsm != 1).values
mbT = [c for c in df.columns if c.startswith("mb_temperature")]; fan = [c for c in df.columns if c.startswith("mb_fan_state")]; bT = [c for c in df.columns if c.startswith("bmsT")]
for nm, cols in [("main-board temperature", mbT), ("fan state", fan), ("BMS temperature", bT)]:
    v = df[cols]; print("  %s columns: %d; non-constant: %s; range %.1f-%.1f" % (nm, len(cols), [c for c in cols if v[c].nunique() > 1], np.nanmin(v.values), np.nanmax(v.values)))
d = df[fresh].copy(); d["off"] = d.P_bms - d.P_mb_f
d["mbT"] = d[[c for c in mbT if df[c].nunique() > 1]].mean(1) if any(df[c].nunique() > 1 for c in mbT) else np.nan
d["fan"] = d[[c for c in fan if df[c].nunique() > 1]].mean(1) if any(df[c].nunique() > 1 for c in fan) else np.nan
d["bmsT"] = d[[c for c in bT if df[c].nunique() > 1]].mean(1) if any(df[c].nunique() > 1 for c in bT) else np.nan
d["uptime_min"] = d.tick / 60000.0; d["Tmot"] = T[fresh].mean(1)
so = d.groupby("segid").agg(boot=("boot", "first"), robot=("robot", "first"), off=("off", "mean"), V=("volt_v", "mean"), soc=("soc", "mean"), P=("P_bms", "mean"), mbT=("mbT", "mean"), fan=("fan", "mean"),
                            bmsT=("bmsT", "mean"), uptime_min=("uptime_min", "mean"), Tmot=("Tmot", "mean"))
cov = [c for c in ["V", "soc", "P", "mbT", "fan", "bmsT", "uptime_min", "Tmot"] if so[c].notna().any() and so[c].nunique() > 1]
ses = so.groupby("boot").mean(numeric_only=True); ses["robot"] = so.groupby("boot").robot.first()
print("per session:"); print(ses[["robot", "off"] + cov].round(2).to_string())
wi = so[["off"] + cov] - so.groupby("boot")[["off"] + cov].transform("mean")
print("correlation of the offset with each quantity:  between sessions (n=%d)   within session (n=%d segments)" % (len(ses), len(so)))
for c in cov: print("   %-11s  %+.2f   %+.2f" % (c, np.corrcoef(ses.off, ses[c])[0, 1], np.corrcoef(wi.off, wi[c])[0, 1]))
print("offset sd between sessions %.2f W; within session %.2f W; robot A %.1f W, robot B %.1f-%.1f W" % (ses.off.std(), wi.off.std(), ses.off[ses.robot == "A"].mean(), ses.off[ses.robot == "B"].min(), ses.off[ses.robot == "B"].max()))
print("pack voltage %.1f-%.1f V, state of charge %.0f-%.0f %% (all frames)" % (df.volt_v.min(), df.volt_v.max(), df.soc.min(), df.soc.max()))

# ================================================================== D. residual diagnostics
print("\n== D. held-out segment error of v2 against operating conditions ==")
p, _ = g.lobo_cv(meta, Xc[BASE], y, boots, **KW)
se = pd.DataFrame({"e": pd.Series(p - y).groupby(segid).mean(), "P": pd.Series(y).groupby(segid).mean()}).join(so[["V", "soc", "uptime_min", "Tmot"]])
for c in ["P", "V", "soc", "uptime_min", "Tmot"]:
    x = se.dropna(subset=[c]); r = stats.linregress(x[c], x.e)
    print("   error vs %-10s slope %+.4f W per unit (95%% %+.4f to %+.4f), r = %+.2f" % (c, r.slope, r.slope - 1.96 * r.stderr, r.slope + 1.96 * r.stderr, r.rvalue))
pe = pd.Series(p - y); hi = y > 250
print("per-sample bias by measured power: below 150 W %+.2f, 150-200 W %+.2f, 200-250 W %+.2f, above 250 W %+.2f W (n above 250 W: %d)" % (
    pe[y < 150].mean(), pe[(y >= 150) & (y < 200)].mean(), pe[(y >= 200) & (y < 250)].mean(), pe[hi].mean(), int(hi.sum())))
print("per-sample bias above 250 W, non-locomotion rows only: %+.2f W (n = %d)" % (pe[hi & ~meta.action.isin(g.LOCO).values].mean(), int((hi & ~meta.action.isin(g.LOCO).values).sum())))

# ================================================================== E. model variants
print("\n== E. model variants (segment-mean error on held-out data, W) ==")
def evaluate(label, cols, kw, extra=()):
    Xs = Xc[cols] if not isinstance(cols, pd.DataFrame) else cols
    p1, _ = g.lobo_cv(meta, Xs, y, boots, **kw); e1 = pd.Series(p1 - y).groupby(segid).mean()
    p2, _ = g.lobo_cv(meta, Xs, y, acts, **kw); e2 = pd.Series(p2 - y).groupby(segid).mean()
    c = g.fit_bounded(Xs[~gest_rows], y[~gest_rows], **kw); e3 = pd.Series((Xs.values @ c.values - y)[gest_rows]).groupby(segid[gest_rows]).mean()
    ca = g.fit_bounded(Xs, y, **kw)
    return dict(model=label, session_mae=e1.abs().mean(), session_max=e1.abs().max(), action_mae=e2.abs().mean(), action_max=e2.abs().max(), nogesture_mae=e3.abs().mean(),
                P0=ca.iloc[0], **{k_: ca[k_] for k_ in extra})
NEG = -np.inf
PAIR = Xc[[c + "@P" for c in BASE]].rename(columns=lambda c: c[:-2])
CU = [c for c in BASE if c.startswith("cu_")]
rows = [evaluate("v2 as in the draft", BASE, KW, ("mech_neg",)),
        evaluate("braking applied to the summed joint power (common bus)", ["const"] + CU + ["netpos", "netneg"], dict(fixed={"netpos": 1.0}, upper={"netneg": 1.0}), ("netneg",)),
        evaluate("parallel joints: mean temperature of the two motors", PAIR, KW, ("mech_neg",)),
        evaluate("+ cross terms tau_pitch*tau_roll (ankle, waist)", BASE + ["x_ankle", "x_waist"], dict(lower={"x_ankle": NEG, "x_waist": NEG}, **KW), ("x_ankle", "x_waist")),
        evaluate("+ |tau| terms (legs, waist, arms)", BASE + ["abs_leg", "abs_waist", "abs_arm"], KW, ("abs_leg", "abs_waist", "abs_arm")),
        evaluate("+ bus resistance term (I^2)", BASE + ["i2"], KW, ("i2",)),
        evaluate("+ absolute temperature level terms (free sign)", BASE + ["lvl_arm", "lvl_ankle", "lvl_other"], dict(lower={"lvl_arm": NEG, "lvl_ankle": NEG, "lvl_other": NEG}, **KW), ("lvl_arm", "lvl_ankle", "lvl_other"))]
RV = pd.DataFrame(rows); print(RV.round(4).to_string(index=False)); RV.round(5).to_csv(os.path.join(A, "rev2_model_variants.csv"), index=False, encoding="utf-8-sig")
cp = g.fit_bounded(PAIR, y, **KW); print("largest change of a coefficient with pair-mean temperatures: %.4f (relative %.2f%%)" % (np.abs(cp.values - coef_all).max(), 100 * np.nanmax(np.abs(cp.values - coef_all) / np.abs(coef_all))))
# implied temperature coefficient of the absolute level, with a session bootstrap
cols = BASE + ["lvl_arm", "lvl_ankle", "lvl_other"]; kw = dict(lower={"lvl_arm": NEG, "lvl_ankle": NEG, "lvl_other": NEG}, **KW)
sess = np.array(sorted(set(boots))); rb = {b: np.where(boots == b)[0] for b in sess}; bs = []
for r in range(200):
    ii = np.concatenate([rb[b] for b in rng.choice(sess, len(sess))]); bs.append(g.fit_bounded(Xc[cols].iloc[ii], y[ii], **kw)[["lvl_arm", "lvl_ankle", "lvl_other"]])
bs = pd.DataFrame(bs); print("implied coefficient of the temperature level (1/K; copper would be +0.0039), 95% by session bootstrap:")
print(pd.DataFrame({"estimate": g.fit_bounded(Xc[cols], y, **kw)[["lvl_arm", "lvl_ankle", "lvl_other"]], "lo": bs.quantile(.025), "hi": bs.quantile(.975)}).round(4).to_string())
print("segment-start temperature range (C): arm motors %.0f-%.0f, ankle motors %.0f-%.0f, other leg and waist motors %.0f-%.0f" % (
    T0[:, g.ARM].min(), T0[:, g.ARM].max(), T0[:, ANK].min(), T0[:, ANK].max(), T0[:, OTH].min(), T0[:, OTH].max()))

# ================================================================== F. gesture decomposition including the temperature factor
print("\n== F. net hold power by gesture with the temperature factor included (W) ==")
kfac = 1.0 + ALPHA * (T - T0); rec = []
for sid, gg in df[df.action.isin(GEST)].groupby("segid", sort=False):
    ix = gg.sort_values("t").index.values; gg = df.loc[ix]; idx = np.arange(len(gg))
    if len(gg) < 150 or (gg.fsm == 1).any(): continue
    dqa = gg.armsp.values; mv = np.where(dqa > 1.0)[0]; hold = gg.arm_holding.fillna(False).astype(bool).values
    if len(mv) == 0 or hold.sum() < 40: continue
    h0 = np.where(hold)[0][0]; h1 = np.where(hold)[0][-1]
    pre = (idx >= 8) & (idx < mv[0] - 5); hd = hold & (dqa < 0.6) & (idx >= h0 + 20) & (idx <= h1 - 3)
    if pre.sum() < 15 or hd.sum() < 20: continue
    c1 = (cu_j[ix] * kfac[ix])[hd].mean(0) - (cu_j[ix] * kfac[ix])[pre].mean(0); c0 = cu_j[ix][hd].mean(0) - cu_j[ix][pre].mean(0)
    rec.append(dict(action=gg.action.iloc[0], meas=gg.P_bms.values[hd].mean() - gg.P_bms.values[pre].mean(), arms=c1[g.ARM].sum(), waist=c1[g.WAIST].sum(), legs=c1[g.LEG].sum(), total_without_factor=c0.sum()))
F = pd.DataFrame(rec); F["total"] = F.arms + F.waist + F.legs
Fg = F.groupby("action").mean(numeric_only=True); Fg["n"] = F.groupby("action").size(); Fg["shortfall"] = Fg.meas - Fg.total; Fg["shortfall_without_factor"] = Fg.meas - Fg.total_without_factor
print(Fg.sort_values("meas").round(2).to_string()); Fg.round(3).to_csv(os.path.join(A, "rev2_gesture_decomposition_with_factor.csv"), encoding="utf-8-sig")
print("all 74 segments: measured %.2f, model with factor %.2f, without factor %.2f W; reject per segment: min %.1f, max %.1f W" % (F.meas.mean(), F.total.mean(), F.total_without_factor.mean(), F.meas[F.action == "reject"].min(), F.meas[F.action == "reject"].max()))

# ================================================================== G. excitation per actuator group
print("\n== G. excitation of each coefficient group (all non-damping frames) ==")
m = (df.fsm != 1).values; rows = []
for gn, types in M["groups"].items():
    js = [j for j, t in enumerate(TYPE) if t in types]; at = np.abs(tau[m][:, js]); aq = np.abs(dq[m][:, js])
    rows.append(dict(group=gn, joints=len(js), tau_rms=np.sqrt((at ** 2).mean()), tau_p99=np.percentile(at, 99), tau_max=at.max(), dq_rms=np.sqrt((aq ** 2).mean()), dq_p99=np.percentile(aq, 99), dq_max=aq.max(),
                     copper_mean_W=(M["b_group"][gn] * (at ** 2).sum(1)).mean(), copper_p99_W=np.percentile(M["b_group"][gn] * (at ** 2).sum(1), 99)))
G = pd.DataFrame(rows).set_index("group"); print(G.round(2).to_string()); G.round(3).to_csv(os.path.join(A, "rev2_excitation.csv"), encoding="utf-8-sig")

# ================================================================== I. details
print("\n== I. details ==")
quiet = (df.speed.values < 0.5) & m & (df.armcu.values < 3.0)
per = df[quiet].groupby("segid").agg(n=("t", "size"), P=("P_bms", "mean"), qa=("qa", "mean"), ankW=("ankW", "mean"), ankT=("Tank", "mean"), robot=("robot", "first"), action=("action", "first"), boot=("boot", "first"))
per = per[(per.n >= 20) & per.qa.notna()]
out = per[(per.qa < thr) & (per.ankW < 3.0)]
print("Fig. 6b: deep-stance points with less than 3 W per ankle motor:"); print(out.round(2).to_string())
for s_ in out.index:
    gg = df[(df.segid == s_) & quiet]; n3 = len(gg) // 3
    print("   %s: ankle angle first third %.1f deg, last third %.1f deg; ankle pitch torque first third %.1f, last third %.1f N*m" % (s_, gg.qa.iloc[:n3].mean(), gg.qa.iloc[-n3:].mean(), tau[gg.index[:n3]][:, [4, 10]].mean(), tau[gg.index[-n3:]][:, [4, 10]].mean()))
print("standing phases used in Fig. 6: %d segments, by action: %s" % (len(per), per.action.value_counts().to_dict()))
