"""Analyses for the third (internal) review of the draft. Nine test sessions only; reads the main line, writes into this folder.
A energy over the moving phase of walking, stepping and turning: held-out error in J and %, per session
B share of the waist and legs in the net power of gestures under refits and alternative model structures
C sensor response re-identified inside the training data for the remaining splits; fixed six-frame window on all four splits
D supplementary segment table with times and the rows used
Usage: python 34_review3_analyses.py <project root>"""
import sys, os, json
import numpy as np, pandas as pd
from scipy.optimize import nnls
ROOT = sys.argv[1]; HERE = os.path.dirname(os.path.abspath(__file__)); MSOUT = os.path.join(os.path.dirname(HERE), "manuscript")
sys.path.insert(0, os.path.join(ROOT, "analysis"))
import g1power as g
pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_rows", 200)
df = g.load(ROOT); M = json.load(open(os.path.join(ROOT, "g1_power_model_v2.json"), encoding="utf-8"))
KW = dict(fixed={"mech_pos": 1.0}, upper={"mech_neg": 1.0})
tau, dq, T = g.arrays(df); pmj = tau * dq
TYPE = [n.replace("left_", "").replace("right_", "") for n in g.JN]; gnames = list(M["groups"].keys()); GRP_OF = {t: gn for gn, types in M["groups"].items() for t in types}
fsm_ok = (df.fsm != 1).values
XT16 = g.features(df, g.V2_SPEC); XT16.insert(0, "const", 1.0); X = g.tie_types(XT16)                       # proposed model
X16 = g.features(df, dict(cu="type", mech="split")); X16.insert(0, "const", 1.0); X0 = g.tie_types(X16)     # without the temperature factor
E = pd.DataFrame(index=df.index); net = pmj.sum(1); E["netpos"] = np.clip(net, 0, None); E["netneg"] = np.clip(net, None, 0)
E["abs_leg"] = np.abs(tau[:, g.LEG]).sum(1); E["abs_waist"] = np.abs(tau[:, g.WAIST]).sum(1); E["abs_arm"] = np.abs(tau[:, g.ARM]).sum(1)
BIG = pd.concat([X, X0.add_suffix("@0"), XT16.add_suffix("@T16"), X16.add_suffix("@16"), E], axis=1)
meta_all, Xcat = g.bms_align(df, BIG); k = (meta_all.fsm != 1).values; meta = meta_all[k]; Xc = Xcat[k]
def sub(suffix, base): return Xc[[c + suffix for c in base]].rename(columns=lambda c: c[:-len(suffix)])
Xf = Xc[list(X.columns)]; Xf0 = sub("@0", list(X0.columns)); XfT16 = sub("@T16", list(XT16.columns)); Xf16 = sub("@16", list(X16.columns))
y = meta.P_bms.values; segid = meta.segid.values; boots = meta.boot.values; acts = meta.action.values; tt = meta.t.values
sess = np.array(sorted(set(boots))); rng = np.random.default_rng(5)
p, C = g.lobo_cv(meta, Xf, y, boots, **KW)                         # prediction of every fresh BMS row with its session held out
es = pd.Series(p - y).groupby(segid).mean(); print("check: held-out session, segment means: MAE %.2f W, largest %.2f W, n = %d" % (es.abs().mean(), es.abs().max(), len(es)))

# ================================================================== A. energy over the moving phase
print("\n== A. moving phase of locomotion segments: energy measured by the BMS and predicted with the session held out ==")
ls = np.abs(dq[:, g.LEG]).sum(1); rows = []
for sid, gg in df[df.action.isin(g.LOCO)].groupby("segid"):
    gg = gg.sort_values("t"); mv = (pd.Series(ls[gg.index]).rolling(10, center=True, min_periods=1).mean() > 1.5).values
    if mv.sum() < 30: continue
    i0, i1 = np.where(mv)[0][0], np.where(mv)[0][-1]; t0, t1 = gg.t.values[i0 + 12], gg.t.values[i1 - 8]          # as in the supplementary table: 1.2 s after the start, 0.8 s before the end
    r = (segid == sid) & (tt >= t0) & (tt <= t1)
    if r.sum() < 4: continue
    dur = t1 - t0; rows.append(dict(segid=sid, action=gg.action.iloc[0], session=int(gg.boot.iloc[0]) + 1, robot=gg.robot.iloc[0], dur_s=dur, bms_samples=int(r.sum()), P_meas=y[r].mean(), P_pred=p[r].mean()))
L = pd.DataFrame(rows); L["E_meas"] = L.P_meas * L.dur_s; L["E_pred"] = L.P_pred * L.dur_s; L["err_J"] = L.E_pred - L.E_meas; L["err_pct"] = 100 * L.err_J / L.E_meas
L["net_meas"] = (L.P_meas - M["P0"]) * L.dur_s; L["err_pct_of_net"] = 100 * L.err_J / L.net_meas
def agg(x): return pd.Series(dict(segments=len(x), sessions=x.session.nunique() if "session" in x else 1, dur_s=x.dur_s.mean(), P_meas=x.P_meas.mean(), E_meas_J=x.E_meas.mean(), mae_J=x.err_J.abs().mean(), mae_pct=x.err_pct.abs().mean(),
                                  bias_J=x.err_J.mean(), bias_pct=x.err_pct.mean(), largest_pct=x.err_pct.abs().max(), mae_W=(x.P_pred - x.P_meas).abs().mean(), mae_pct_of_net=x.err_pct_of_net.abs().mean()))
LA = pd.concat([L.groupby("action").apply(agg), pd.DataFrame({"all locomotion": agg(L)}).T]); print(LA.round(2).to_string())
print("per session:"); print(L.groupby("session").apply(agg)[["segments", "E_meas_J", "mae_J", "mae_pct", "bias_J", "bias_pct"]].round(2).to_string())
L.round(3).to_csv(os.path.join(HERE, "review3_locomotion_energy.csv"), index=False, encoding="utf-8-sig"); LA.round(3).to_csv(os.path.join(HERE, "review3_locomotion_energy_summary.csv"), encoding="utf-8-sig")
ybar = pd.Series(y).groupby(segid).mean(); info = meta.groupby("segid").agg(action=("action", "first")); pct = 100 * es / ybar; loco_s = info.action.isin(g.LOCO)
print("whole segments (energy error in %% = error of the mean power): all %d segments mean |error| %.2f%%, largest %.2f%%; locomotion segments (%d) %.2f%%; other segments %.2f%%" % (
    len(pct), pct.abs().mean(), pct.abs().max(), int(loco_s.sum()), pct[loco_s].abs().mean(), pct[~loco_s].abs().mean()))

# ================================================================== B. share of waist and legs
print("\n== B. net power of gestures assigned to waist and legs, under refits and alternative structures ==")
GEST = ["hands up", "heart", "high five", "hug", "reject", "shake hand", "xray"]
def phases(gg):
    if len(gg) < 150 or (gg.fsm == 1).any(): return None, None
    dqa = gg[[f"dq{j}" for j in g.ARM]].abs().sum(1).values; mv = np.where(dqa > 1.0)[0]; hold = gg.arm_holding.fillna(False).astype(bool).values
    if len(mv) == 0 or hold.sum() < 40: return None, None
    h0 = np.where(hold)[0][0]; h1 = np.where(hold)[0][-1]; idx = np.arange(len(gg))
    pre = (idx >= 8) & (idx < mv[0] - 5); hd = hold & (dqa < 0.6) & (idx >= h0 + 20) & (idx <= h1 - 3)
    if pre.sum() < 15 or hd.sum() < 20: return None, None
    return pre, hd
recs = []
for sid, gg in df[df.action.isin(GEST)].groupby("segid", sort=False):
    gg = gg.sort_values("t").reset_index(drop=True); pre, hd = phases(gg)
    if pre is None: continue
    Tg = gg[[f"T{j}" for j in range(29)]].values.astype(float); tq = gg[[f"tau{j}" for j in range(29)]].values.astype(float); t2 = tq ** 2; t2T = t2 * (1.0 + g.ALPHA_CU * (Tg - Tg[:15].mean(0)))
    recs.append(dict(segid=sid, action=gg.action.iloc[0], boot=int(gg.boot.iloc[0]), meas=gg.P_bms.values[hd].mean() - gg.P_bms.values[pre].mean(),
                     d=t2T[hd].mean(0) - t2T[pre].mean(0), d0=t2[hd].mean(0) - t2[pre].mean(0), da=np.abs(tq)[hd].mean(0) - np.abs(tq)[pre].mean(0)))
P = pd.DataFrame(recs); D = np.array(P.d.tolist()); D0 = np.array(P.d0.tolist()); DA = np.array(P.da.tolist()); meas = P.meas.values
bj = np.array([M["b"][t] for t in TYPE]); cl = (D * bj)[:, 15:22].sum(1); cr = (D * bj)[:, 22:29].sum(1)
weak = pd.DataFrame({"a": P.action, "l": cl, "r": cr}).groupby("a").mean(); weak = weak.min(1) / weak.sum(1); TWO = weak.index[weak >= 0.25].tolist(); two = P.action.isin(TWO).values
print("net-power set: %d segments; two-arm gestures %s: %d segments, measured net power %.1f W" % (len(P), TWO, int(two.sum()), meas[two].mean()))
by_group = lambda c: np.array([c["cu_" + GRP_OF[t]] for t in TYPE]); by_type = lambda c: np.array([c["cu_" + t] for t in TYPE])
WL = g.WAIST + g.LEG
def decomp(label, bvec, Dm, add=None, mask=two):
    c = Dm[mask] * bvec; a_, w_, l_ = c[:, g.ARM].sum(1), c[:, g.WAIST].sum(1), c[:, g.LEG].sum(1)
    if add is not None: a_, w_, l_ = a_ + add[0][mask], w_ + add[1][mask], l_ + add[2][mask]
    return dict(variant=label, arms=a_.mean(), waist=w_.mean(), legs=l_.mean(), model_total=(a_ + w_ + l_).mean(), measured=meas[mask].mean(),
                share_of_measured=100 * (w_ + l_).sum() / meas[mask].sum(), share_of_model_total=100 * (w_ + l_).sum() / (a_ + w_ + l_).sum())
V = [decomp("proposed model, all data", bj, D)]
# leave-one-session-out refits
lo = [decomp("fit without session %d" % (b + 1), by_group(C[b]), D) for b in sess]; sh = [r["share_of_measured"] for r in lo]
print("refits without one session: share of measured %.1f-%.1f%% (the nine fits: %s)" % (min(sh), max(sh), ", ".join("%.1f" % v for v in sh)))
ho = np.zeros((len(P), 3))
for b in sess:
    mk = (P.boot.values == b); cc = D[mk] * by_group(C[b]); ho[mk] = np.column_stack([cc[:, g.ARM].sum(1), cc[:, g.WAIST].sum(1), cc[:, g.LEG].sum(1)])
V.append(dict(variant="each segment decomposed with coefficients fitted without its session", arms=ho[two, 0].mean(), waist=ho[two, 1].mean(), legs=ho[two, 2].mean(), model_total=ho[two].sum(1).mean(), measured=meas[two].mean(),
              share_of_measured=100 * ho[two, 1:].sum() / meas[two].sum(), share_of_model_total=100 * ho[two, 1:].sum() / ho[two].sum()))
V.append(dict(variant="range over the nine fits without one session", share_of_measured=np.nan, arms=np.nan, waist=np.nan, legs=np.nan, model_total=np.nan, measured=meas[two].mean(), share_of_model_total=np.nan, lo=min(sh), hi=max(sh)))
# alternative structures, all data
c0 = g.fit_bounded(Xf0, y, **KW); V.append(decomp("shared coefficients, no temperature factor", by_group(c0), D0))
c16 = g.fit_bounded(Xf16, y, **KW); V.append(decomp("one coefficient per joint type, no temperature factor", by_type(c16), D0))
cT16 = g.fit_bounded(XfT16, y, **KW); V.append(decomp("one coefficient per joint type, with temperature factor", by_type(cT16), D))
CU = [c for c in Xf.columns if c.startswith("cu_")]
cb = g.fit_bounded(Xc[["const"] + CU + ["netpos", "netneg"]], y, fixed={"netpos": 1.0}, upper={"netneg": 1.0}); V.append(decomp("braking applied to the summed joint power", by_group(cb), D))
ca = g.fit_bounded(Xc[list(Xf.columns) + ["abs_leg", "abs_waist", "abs_arm"]], y, **KW)
V.append(decomp("with terms linear in |torque|", by_group(ca), D, add=(ca["abs_arm"] * DA[:, g.ARM].sum(1), ca["abs_waist"] * DA[:, g.WAIST].sum(1), ca["abs_leg"] * DA[:, g.LEG].sum(1))))
K = len(g.BMS_FIR) - 1
df["P_mb_f"] = df.groupby("segid").P_mb.transform(lambda s: np.convolve(s.values, g.BMS_FIR, mode="full")[:len(s)]); ymb = df.P_mb_f.values[meta.index.values]; okm = np.isfinite(ymb)
Xmb = Xf.drop(columns="const").copy()
for b in sess: Xmb["s%d" % b] = (boots == b).astype(float)
cm = g.fit_bounded(Xmb[okm], ymb[okm], lower={"s%d" % b: -np.inf for b in sess}, **KW); V.append(decomp("main-board power as the target, one intercept per session", by_group(cm), D))
# fits in the net-power framing: gesture segments only, no constant term, no other motions
Dg = pd.DataFrame({"cu_" + gn: D[:, [j for j, t in enumerate(TYPE) if GRP_OF[t] == gn]].sum(1) for gn in gnames}); cn = g.fit_bounded(Dg, meas); V.append(decomp("fitted to the net power of the 74 gesture segments only, shared coefficients", by_group(cn), D))
Dt = pd.DataFrame({"cu_" + t: D0[:, [j for j, tt_ in enumerate(TYPE) if tt_ == t]].sum(1) for t in g.TYPES}); cnt = g.fit_bounded(Dt, meas); V.append(decomp("fitted to the net power of the 74 gesture segments only, one coefficient per joint type", by_type(cnt), D0))
en = np.zeros(len(P))
for b in sorted(set(P.boot)):
    mk = (P.boot.values == b); en[mk] = Dg[mk].values @ g.fit_bounded(Dg[~mk], meas[~mk]).values - meas[mk]
print("net-power fit with shared coefficients, session held out: MAE %.2f W, largest %.2f W (74 segments)" % (np.abs(en).mean(), np.abs(en).max()))
VT = pd.DataFrame(V); print(VT.round(2).to_string(index=False)); VT.round(3).to_csv(os.path.join(HERE, "review3_share_variants.csv"), index=False, encoding="utf-8-sig")
alt = VT[VT.share_of_measured.notna()]; print("share of measured over all variants: %.1f-%.1f%%; share of the model total: %.1f-%.1f%%" % (alt.share_of_measured.min(), alt.share_of_measured.max(), alt.share_of_model_total.min(), alt.share_of_model_total.max()))
# what the interval of the decomposition contains: coefficient sets from resampling segments (stored) and from resampling sessions (here)
S = np.array(M["uncertainty"]["bootstrap_samples"]); BJ = np.array([[S[r, 1 + gnames.index(GRP_OF[t])] for t in TYPE] for r in range(len(S))])
def shares(BJm):
    c = D[two][None, :, :] * BJm[:, None, :]; return 100 * c[:, :, g.WAIST].sum((1, 2)) / meas[two].sum(), 100 * c[:, :, g.LEG].sum((1, 2)) / meas[two].sum()
w_s, l_s = shares(BJ); print("segment-resampled coefficient sets (300), segments fixed: waist share %.1f%% (sd %.1f), leg share %.1f%% (sd %.1f), sum sd %.1f, correlation of the two %.2f; 95%% of the sum %.1f-%.1f%%" % (
    w_s.mean(), w_s.std(), l_s.mean(), l_s.std(), (w_s + l_s).std(), np.corrcoef(w_s, l_s)[0, 1], *np.quantile(w_s + l_s, [.025, .975])))
gi = {gn: 1 + i for i, gn in enumerate(gnames)}; print("correlation of the N7520-14.3 and waist-pitch coefficients across those sets: %.2f" % np.corrcoef(S[:, gi["N7520_14"]], S[:, gi["waist_pitch"]])[0, 1])
rows_b = {b: np.where(boots == b)[0] for b in sess}; Xr = Xf.reset_index(drop=True); BS = []
for r in range(300):
    ii = np.concatenate([rows_b[b] for b in rng.choice(sess, len(sess))]); BS.append(by_group(g.fit_bounded(Xr.iloc[ii], y[ii], **KW)))
w_b, l_b = shares(np.array(BS)); print("session-resampled coefficient sets (300), segments fixed: sum %.1f%% (95%% %.1f-%.1f%%); waist %.1f-%.1f%%, legs %.1f-%.1f%%" % (
    (w_b + l_b).mean(), *np.quantile(w_b + l_b, [.025, .975]), *np.quantile(w_b, [.025, .975]), *np.quantile(l_b, [.025, .975])))
for a_ in TWO:
    mk = (P.action == a_).values; r1 = decomp(a_, bj, D, mask=mk); r2 = decomp(a_, by_type(c16), D0, mask=mk); r3 = decomp(a_, by_group(cn), D, mask=mk)
    print("   %-9s share of measured: proposed %.0f%%, per-joint-type %.0f%%, net-power fit %.0f%%" % (a_, r1["share_of_measured"], r2["share_of_measured"], r3["share_of_measured"]))

# ================================================================== C. sensor response on the remaining splits
print("\n== C. sensor handling on all four splits (segment-mean error, MAE / largest, W) ==")
def est_kernel(mask, Kk=8):
    rows_ = []; dm = df[mask]
    for sid, idx in dm.groupby("segid", sort=False).indices.items():
        gg = dm.iloc[idx]; pm_ = gg.P_mb.interpolate().bfill().values; pb = gg.P_bms.values; I = gg.cur_raw.values; t = gg.t.values
        chg = np.r_[False, np.diff(I) != 0]; okk = np.r_[True, np.diff(t) < 0.2]
        for i in np.where(chg)[0]:
            if i < Kk or not okk[i - Kk:i + 1].all() or not np.isfinite(pm_[i - Kk:i + 1]).all(): continue
            rows_.append((gg.boot.iloc[0], pb[i]) + tuple(pm_[i - kk] for kk in range(Kk + 1)))
    R = pd.DataFrame(rows_); Xk = R.iloc[:, 2:].values - R.groupby(0)[list(R.columns[2:])].transform("mean").values; yk = R[1].values - R.groupby(0)[1].transform("mean").values
    h, _ = nnls(Xk, yk); return h / h.sum()
def aligned(fir):
    ma, Xa = g.bms_align(df, X, fir=fir); kq = (ma.fsm != 1).values; return ma[kq], Xa[kq]
def seg_err(ma, Xa, tr, te):
    yy = ma.P_bms.values; c = g.fit_bounded(Xa[tr], yy[tr], **KW); return pd.Series((Xa.values @ c.values - yy)[te]).groupby(ma.segid.values[te]).mean()
def four_splits(fir):
    ma, Xa = aligned(fir); out = {}
    for name, grp in [("session", ma.boot.values), ("action class", ma.action.values)]:
        e = pd.concat([seg_err(ma, Xa, grp != v, grp == v) for v in pd.unique(grp)]); out[name] = (e.abs().mean(), e.abs().max())
    gest = ~ma.action.isin(g.LOCO + ["stand still", "squat"]).values; e = seg_err(ma, Xa, ~gest, gest); out["no gestures"] = (e.abs().mean(), e.abs().max())
    loco = ma.action.isin(g.LOCO).values; e = seg_err(ma, Xa, ~loco, loco); out["no locomotion"] = (e.abs().mean(), e.abs().max())
    return out
box = np.zeros(7); box[1:7] = 1 / 6.0
res = {"identified response (all data)": four_splits(g.BMS_FIR), "window over lags 1-6, fixed": four_splits(box)}
nest = {}
e = []
for a_ in sorted(pd.unique(acts)):
    ma, Xa = aligned(est_kernel((df.action != a_).values & fsm_ok)); te = (ma.action == a_).values
    if te.any(): e.append(seg_err(ma, Xa, ~te, te))
e = pd.concat(e); nest["action class"] = (e.abs().mean(), e.abs().max())
kl = est_kernel(~df.action.isin(g.LOCO).values & fsm_ok); ma, Xa = aligned(kl); loco = ma.action.isin(g.LOCO).values; e = seg_err(ma, Xa, ~loco, loco); nest["no locomotion"] = (e.abs().mean(), e.abs().max())
print("response identified without locomotion data:", np.round(kl, 3))
res["response re-identified inside the training data"] = nest
KT = pd.DataFrame({k_: {s_: "%.2f / %.1f" % v for s_, v in d_.items()} for k_, d_ in res.items()}).T.reindex(columns=["session", "action class", "no gestures", "no locomotion"]); print(KT.to_string())
KT.to_csv(os.path.join(HERE, "review3_kernel_splits.csv"), encoding="utf-8-sig")
print("(re-identified response for the held-out session and the no-gesture split: 1.65 / 5.9 and 4.7 / 11.5, main line, 23_review_checks.py)")

# ================================================================== D. supplementary table with times
sup = pd.read_csv(os.path.join(ROOT, "manuscript", "supplement_segments.csv")); key = sup.date + "|" + sup.segment
t0 = df.groupby("segid").t.min(); t1 = df.groupby("segid").t.max(); iso = lambda u: pd.Timestamp(u, unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%S.%f")[:-5] + "Z"
sup["start_utc"] = key.map(t0).map(iso); sup["end_utc"] = key.map(t1).map(iso)
fr = pd.DataFrame({"s": segid, "t": tt}).groupby("s").t.agg(["min", "max"])
sup["first_row_used_s"] = (key.map(fr["min"]) - key.map(t0)).round(2); sup["last_row_used_s"] = (key.map(fr["max"]) - key.map(t0)).round(2)
sup.to_csv(os.path.join(MSOUT, "supplement_S1_segments.csv"), index=False, encoding="utf-8-sig")
print("\n== D. supplementary table S1: %d segments, %d with rows in the fit; first row used %.2f-%.2f s after the start of the segment ==" % (len(sup), int(sup.first_row_used_s.notna().sum()), sup.first_row_used_s.min(), sup.first_row_used_s.max()))
