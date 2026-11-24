"""Checks and additions for the fourth (internal) review round. Reads the main line, writes into this folder.
A difference between the two power sensors per session, with two ways of averaging
B held-out action class and no-gesture split for three structures: what sharing alone gives, what the temperature factor adds
C net power of the 74 gesture holds: per-joint-type and shared coefficients fitted to net power, and the ten-parameter model
D energy of the moving window: how repeated BMS readings and unequal intervals enter; equal weights against time weights
E Supplementary Table S1 with per-segment results
Usage: python 35_review4_checks.py <project root>"""
import sys, os, json
import numpy as np, pandas as pd
ROOT = sys.argv[1]; HERE = os.path.dirname(os.path.abspath(__file__)); MSOUT = os.path.join(os.path.dirname(HERE), "manuscript")
sys.path.insert(0, os.path.join(ROOT, "analysis"))
import g1power as g
pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_rows", 200)
df = g.load(ROOT); M = json.load(open(os.path.join(ROOT, "g1_power_model_v2.json"), encoding="utf-8"))
KW = dict(fixed={"mech_pos": 1.0}, upper={"mech_neg": 1.0})
tau, dq, T = g.arrays(df); pmj = tau * dq
TYPE = [n.replace("left_", "").replace("right_", "") for n in g.JN]; gnames = list(M["groups"].keys()); GRP_OF = {t: gn for gn, types in M["groups"].items() for t in types}
segi = df.groupby("segid", sort=False).cumcount().values; K = len(g.BMS_FIR) - 1
XT16 = g.features(df, g.V2_SPEC); XT16.insert(0, "const", 1.0); X = g.tie_types(XT16)
X16 = g.features(df, dict(cu="type", mech="split")); X16.insert(0, "const", 1.0); X0 = g.tie_types(X16)
meta_all, Xcat = g.bms_align(df, pd.concat([X, X0.add_suffix("@0"), X16.add_suffix("@16")], axis=1)); k = (meta_all.fsm != 1).values; meta = meta_all[k]; Xc = Xcat[k]
sub = lambda suffix, base: Xc[[c + suffix for c in base]].rename(columns=lambda c: c[:-len(suffix)])
Xf = Xc[list(X.columns)]; Xf0 = sub("@0", list(X0.columns)); Xf16 = sub("@16", list(X16.columns))
y = meta.P_bms.values; segid = meta.segid.values; boots = meta.boot.values; acts = meta.action.values; tt = meta.t.values
sess = np.array(sorted(set(boots)))
p, C = g.lobo_cv(meta, Xf, y, boots, **KW); es = pd.Series(p - y).groupby(segid).mean(); ybar = pd.Series(y).groupby(segid).mean(); pbar = pd.Series(p).groupby(segid).mean()
print("check: held-out session, segment means: MAE %.2f W, largest %.2f W, n = %d" % (es.abs().mean(), es.abs().max(), len(es)))

# ================================================================== A. sensor difference per session, two ways of averaging
print("\n== A. BMS minus main-board power per session (W) ==")
df["P_mb_f"] = df.groupby("segid").P_mb.transform(lambda s: np.convolve(s.values, g.BMS_FIR, mode="full")[:len(s)]); df.loc[segi < K, "P_mb_f"] = np.nan
fresh = df.groupby("segid").cur_raw.transform(lambda s: np.r_[False, np.diff(s.values) != 0]).values.astype(bool) & df.P_mb_f.notna().values & (df.fsm != 1).values
d = df[fresh].copy(); d["off"] = d.P_bms - d.P_mb_f
OA = pd.DataFrame({"robot": d.groupby("boot").robot.first(), "mean_over_samples": d.groupby("boot").off.mean(), "mean_of_segment_means": d.groupby(["boot", "segid"]).off.mean().groupby("boot").mean(), "segments": d.groupby("boot").segid.nunique()})
OA.index = OA.index + 1; OA.index.name = "session"; print(OA.round(2).to_string())
for c in ["mean_over_samples", "mean_of_segment_means"]:
    b_ = OA[OA.robot == "B"][c]; print("   %-22s: all sessions %.1f-%.1f W, sd %.1f; robot A %.1f; robot B %.1f-%.1f" % (c, OA[c].min(), OA[c].max(), OA[c].std(), OA[OA.robot == "A"][c].iloc[0], b_.min(), b_.max()))

# ================================================================== B. what sharing alone gives
print("\n== B. three structures on the splits that test extrapolation (segment-mean error, W) ==")
gest = ~meta.action.isin(g.LOCO + ["stand still", "squat"]).values; rows = []
for label, Xs in [("one coefficient per joint type, no temperature factor", Xf16), ("shared coefficients, no temperature factor", Xf0), ("proposed: shared coefficients and temperature factor", Xf)]:
    pa, _ = g.lobo_cv(meta, Xs, y, acts, **KW); ea = pd.Series(pa - y).groupby(segid).mean(); cls = pd.Series(acts).groupby(segid).first(); bycls = ea.groupby(cls).mean(); worst = ea.abs().groupby(cls).max().idxmax()
    c = g.fit_bounded(Xs[~gest], y[~gest], **KW); eg = pd.Series((Xs.values @ c.values - y)[gest]).groupby(segid[gest]).mean()
    ps, _ = g.lobo_cv(meta, Xs, y, boots, **KW); e1 = pd.Series(ps - y).groupby(segid).mean()
    rows.append(dict(structure=label, session_mae=e1.abs().mean(), session_max=e1.abs().max(), class_mae=ea.abs().mean(), class_max=ea.abs().max(), class_of_largest=worst, reject_bias=bycls["reject"],
                     largest_class_bias=bycls.abs().max(), class_with_largest_bias=bycls.abs().idxmax(), nogesture_mae=eg.abs().mean(), nogesture_max=eg.abs().max()))
print(pd.DataFrame(rows).round(2).to_string(index=False))

# ================================================================== C. net power of the gesture holds
print("\n== C. net power during the hold, 74 gesture segments, session held out (W) ==")
GEST = ["hands up", "heart", "high five", "hug", "reject", "shake hand", "xray"]; parts = []
for sid, gg in df[df.action.isin(GEST)].groupby("segid", sort=False):
    gg = gg.sort_values("t").reset_index(drop=True)
    if len(gg) < 150 or (gg.fsm == 1).any(): continue
    dqa = gg[[f"dq{j}" for j in g.ARM]].abs().sum(1).values; mv = np.where(dqa > 1.0)[0]; hold = gg.arm_holding.fillna(False).astype(bool).values
    if len(mv) == 0 or hold.sum() < 40: continue
    h0 = np.where(hold)[0][0]; h1 = np.where(hold)[0][-1]; idx = np.arange(len(gg))
    gg["is_pre"] = (idx >= 8) & (idx < mv[0] - 5); gg["is_hold"] = hold & (dqa < 0.6) & (idx >= h0 + 20) & (idx <= h1 - 3)
    if gg.is_pre.sum() < 15 or gg.is_hold.sum() < 20: continue
    parts.append(gg)
dg = pd.concat(parts, ignore_index=True); tg, dqg, Tg = g.arrays(dg); sg = dg.segid.values; pre = dg.is_pre.values; hd = dg.is_hold.values
T0g = pd.DataFrame(np.where((dg.groupby("segid", sort=False).cumcount().values < 15)[:, None], Tg, np.nan)).groupby(sg).transform("mean").values
t2 = tg ** 2; t2T = t2 * (1.0 + g.ALPHA_CU * (Tg - T0g)); pmg = tg * dqg
def minus_pre(v): return v - pd.DataFrame(np.where(pre[:, None], v, np.nan)).groupby(sg).transform("mean").values
d2 = minus_pre(t2); d2T = minus_pre(t2T); dg["y_net"] = dg.P_bms.values - dg.P_bms.where(dg.is_pre).groupby(dg.segid).transform("mean").values
Xn = pd.DataFrame(index=dg.index)
for t_, ix in g.TYPE_IDX.items(): Xn["dcu_" + t_] = d2[:, ix].sum(1)
for gn in gnames:
    js = [j for j, t_ in enumerate(TYPE) if GRP_OF[t_] == gn]; Xn["dcuG_" + gn] = d2[:, js].sum(1); Xn["dcuGT_" + gn] = d2T[:, js].sum(1)
Xn["mech_pos"] = np.clip(pmg, 0, None).sum(1); Xn["mech_neg"] = np.clip(pmg, None, 0).sum(1)
mn, Xnf = g.bms_align(dg, Xn); yn = mn.y_net.values; hn = mn.is_hold.values
def hold_stats(label, cols, params):
    pr, _ = g.lobo_cv(mn, Xnf[cols], yn, mn.boot.values, **KW); e = pd.Series((pr - yn)[hn]).groupby(mn.segid.values[hn]).mean()
    return dict(model=label, parameters=params, bias=e.mean(), mae=e.abs().mean(), largest=e.abs().max(), segments=len(e)), e
MP = ["mech_pos", "mech_neg"]; NT = []
r, e_type = hold_stats("whole-body copper terms, one coefficient per joint type, fitted to the net power of the gesture segments", ["dcu_" + t_ for t_ in g.TYPES] + MP, 17); NT.append(r)
r, _ = hold_stats("whole-body copper terms, coefficients shared by actuator, fitted to the net power of the gesture segments", ["dcuG_" + gn for gn in gnames] + MP, 9); NT.append(r)
r, _ = hold_stats("the same with the temperature factor", ["dcuGT_" + gn for gn in gnames] + MP, 9); NT.append(r)
# the ten-parameter model as identified: fitted to the total power of all motions without the session, applied to hold minus standing
seg_of = dg.groupby("segid", sort=False).agg(boot=("boot", "first"), action=("action", "first"))
hm = lambda v, m: pd.DataFrame(v[m]).groupby(sg[m]).mean()
Dseg = hm(t2T, hd) - hm(t2T, pre); mp_ = np.clip(pmg, 0, None).sum(1)[:, None]; mng = np.clip(pmg, None, 0).sum(1)[:, None]
dpos = (hm(mp_, hd) - hm(mp_, pre))[0]; dneg = (hm(mng, hd) - hm(mng, pre))[0]; meas = (hm(dg.P_bms.values[:, None], hd) - hm(dg.P_bms.values[:, None], pre))[0]
pred = pd.Series(index=Dseg.index, dtype=float)
for s_ in Dseg.index:
    c = C[seg_of.boot[s_]]; bvec = np.array([c["cu_" + GRP_OF[t_]] for t_ in TYPE]); pred[s_] = Dseg.loc[s_].values @ bvec + dpos[s_] + c["mech_neg"] * dneg[s_]
e10 = pred - meas; NT.append(dict(model="proposed ten-parameter model, fitted to the total power of all motions", parameters=10, bias=e10.mean(), mae=e10.abs().mean(), largest=e10.abs().max(), segments=len(e10)))
NTt = pd.DataFrame(NT); print(NTt.round(2).to_string(index=False)); NTt.round(3).to_csv(os.path.join(HERE, "review4_net_power_models.csv"), index=False, encoding="utf-8-sig")
print("measured net power: mean %.1f W over %d segments; mechanical terms in the hold-minus-standing difference: mean %.2f W, largest %.2f W" % (meas.mean(), len(meas), (dpos + dneg).abs().mean(), (dpos + dneg).abs().max()))
print("ten-parameter model by gesture (bias, W):", e10.groupby(seg_of.action).mean().round(1).to_dict())
# the parameter beyond the copper-loss coefficients in the net-power fits: the braking coefficient (driving term fixed at one, no constant)
pn = mn.is_pre.values
for label, cols in [("one coefficient per joint type: 16 copper-loss coefficients + braking coefficient", ["dcu_" + t_ for t_ in g.TYPES] + MP), ("shared, with temperature factor: 8 copper-loss coefficients + braking coefficient", ["dcuGT_" + gn for gn in gnames] + MP)]:
    c = g.fit_bounded(Xnf[cols], yn, **KW); _, Cf = g.lobo_cv(mn, Xnf[cols], yn, mn.boot.values, **KW)
    mech = Xnf.mech_pos.values + c["mech_neg"] * Xnf.mech_neg.values; sh_ = pd.Series(mech[hn]).groupby(mn.segid.values[hn]).mean(); sp_ = pd.Series(mech[pn]).groupby(mn.segid.values[pn]).mean()
    c0 = g.fit_bounded(Xnf[cols], yn, fixed={"mech_pos": 1.0, "mech_neg": 0.0}); e0 = pd.Series((Xnf[cols].values @ c0.values - yn)[hn]).groupby(mn.segid.values[hn]).mean()
    print("   %s: braking coefficient %.2f on all gesture data (%.2f-%.2f over the session folds); rows in the fit %d, of which in the hold %d; mechanical terms in the hold: mean %.2f W, largest segment %.2f W; in the standing before: mean %.2f W, largest %.2f W; hold error with the braking coefficient set to zero (in-sample): MAE %.2f W" % (
        label, c["mech_neg"], Cf.loc["mech_neg"].min(), Cf.loc["mech_neg"].max(), len(yn), int(hn.sum()), sh_.abs().mean(), sh_.abs().max(), sp_.abs().mean(), sp_.abs().max(), e0.abs().mean()))

# ================================================================== D. energy of the moving window
print("\n== D. energy of the moving window: equal weights against time weights ==")
ls = np.abs(dq[:, g.LEG]).sum(1); rows = []; gaps = []
for sid, gg in df[df.action.isin(g.LOCO)].groupby("segid"):
    gg = gg.sort_values("t"); mv = (pd.Series(ls[gg.index]).rolling(10, center=True, min_periods=1).mean() > 1.5).values
    r0 = dict(segid=sid, action=gg.action.iloc[0], session=int(gg.boot.iloc[0]) + 1)
    if mv.sum() < 30: rows.append(dict(r0, in_table=0, reason="moving phase shorter than 3 s")); continue
    i0, i1 = np.where(mv)[0][0], np.where(mv)[0][-1]; t0, t1 = gg.t.values[i0 + 12], gg.t.values[i1 - 8]; r0.update(t0=t0 - gg.t.values[0], t1=t1 - gg.t.values[0], dur_s=t1 - t0)
    rs = np.where(segid == sid)[0]; ts = tt[rs]; inw = (ts >= t0) & (ts <= t1); r0["bms_samples"] = int(inw.sum())
    if inw.sum() < 4: rows.append(dict(r0, in_table=0, reason="fewer than four BMS samples in the window")); continue
    nxt = np.r_[ts[1:], np.inf]; w = np.clip(np.minimum(nxt, t1) - np.maximum(ts, t0), 0, None)          # each reading counts from its own time to the next reading, cut at the window edges
    gaps += np.diff(ts[inw]).tolist()
    fr = gg[(gg.t >= t0) & (gg.t <= t1)]; dtf = np.r_[np.diff(fr.t.values), 0.0]
    rows.append(dict(r0, in_table=1, reason="", P_meas=y[rs][inw].mean(), P_pred=p[rs][inw].mean(), E_meas=y[rs][inw].mean() * (t1 - t0), E_pred=p[rs][inw].mean() * (t1 - t0),
                     E_meas_tw=(w * y[rs]).sum(), E_pred_tw=(w * p[rs]).sum(), covered_s=w.sum(), E_meas_frames=(fr.P_bms.values * dtf).sum(), frames=len(fr)))
L = pd.DataFrame(rows); Lt = L[L.in_table == 1].copy()
for sfx in ["", "_tw"]:
    Lt["err" + sfx] = Lt["E_pred" + sfx] - Lt["E_meas" + sfx]; Lt["pct" + sfx] = 100 * Lt["err" + sfx] / Lt["E_meas" + sfx]
def summ(x): return pd.Series(dict(segments=len(x), dur_s=x.dur_s.mean(), E_meas=x.E_meas.mean(), mae_J=x.err.abs().mean(), mae_pct=x.pct.abs().mean(), bias_pct=x.pct.mean(), largest_pct=x.pct.abs().max(),
                                   E_meas_tw=x.E_meas_tw.mean(), mae_J_tw=x.err_tw.abs().mean(), mae_pct_tw=x.pct_tw.abs().mean(), bias_pct_tw=x.pct_tw.mean(), largest_pct_tw=x.pct_tw.abs().max()))
print(pd.concat([Lt.groupby("action").apply(summ), pd.DataFrame({"all": summ(Lt)}).T]).round(2).to_string())
gaps = np.array(gaps); print("time between consecutive BMS readings inside the windows: mean %.3f s, sd %.3f, min %.2f, max %.2f; frames per reading %.1f" % (gaps.mean(), gaps.std(), gaps.min(), gaps.max(), Lt.frames.sum() / Lt.bms_samples.sum()))
print("window covered by the time weights: %.1f-%.1f%% of its duration; measured energy, time weights against equal weights: mean difference %+.2f%%, largest %.2f%%; summing the logged frame values gives %+.2f%% against time weights" % (
    100 * (Lt.covered_s / Lt.dur_s).min(), 100 * (Lt.covered_s / Lt.dur_s).max(), 100 * (Lt.E_meas_tw / Lt.E_meas - 1).mean(), 100 * (Lt.E_meas_tw / Lt.E_meas - 1).abs().max(), 100 * (Lt.E_meas_frames / Lt.E_meas_tw - 1).mean()))
print("left out of the table:"); print(L[L.in_table == 0][["segid", "action", "session", "dur_s", "bms_samples", "reason"]].to_string(index=False))

# ================================================================== E. Supplementary Table S1 with per-segment results
sup = pd.read_csv(os.path.join(ROOT, "manuscript", "supplement_segments.csv")); key = sup.date + "|" + sup.segment
t0s = df.groupby("segid").t.min(); t1s = df.groupby("segid").t.max(); iso = lambda u: pd.Timestamp(u, unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%S.%f")[:-5] + "Z"
sup["start_utc"] = key.map(t0s).map(iso); sup["end_utc"] = key.map(t1s).map(iso)
fr_ = pd.DataFrame({"s": segid, "t": tt}).groupby("s").t.agg(["min", "max"])
sup["first_row_used_s"] = (key.map(fr_["min"]) - key.map(t0s)).round(2); sup["last_row_used_s"] = (key.map(fr_["max"]) - key.map(t0s)).round(2)
sup["mean_power_measured_W"] = key.map(ybar).round(2); sup["mean_power_predicted_session_held_out_W"] = key.map(pbar).round(2); sup["error_W"] = key.map(es).round(2)
sup["hold_net_power_measured_W"] = key.map(meas).round(2); sup["hold_net_power_predicted_W"] = key.map(pred).round(2)
Lk = L.set_index("segid")
sup["moving_window_bms_readings"] = key.map(Lk.bms_samples); sup["in_energy_table"] = key.map(Lk.in_table); sup["energy_table_exclusion"] = key.map(Lk.reason).fillna("")
Ltk = Lt.set_index("segid")
for c_, name in [("E_meas_tw", "window_energy_measured_J"), ("E_pred_tw", "window_energy_predicted_J"), ("err_tw", "window_energy_error_J"), ("pct_tw", "window_energy_error_pct")]: sup[name] = key.map(Ltk[c_]).round(2)
sup.to_csv(os.path.join(MSOUT, "supplement_S1_segments.csv"), index=False, encoding="utf-8-sig"); sup.to_excel(os.path.join(MSOUT, "supplement_S1_segments.xlsx"), index=False)
print("\n== E. Supplementary Table S1: %d segments, %d columns; segments with a prediction %d; in the net-power set %d; in the energy table %d ==" % (len(sup), sup.shape[1], int(sup.error_W.notna().sum()), int(sup.hold_net_power_measured_W.notna().sum()), int((sup.in_energy_table == 1).sum())))
print("from S1 alone: MAE of the segment mean %.2f W; mean of the session MAEs %.2f W; window energy MAE %.1f J, %.2f%%" % (sup.error_W.abs().mean(), sup.groupby("session").error_W.apply(lambda s: s.abs().mean()).mean(), sup.window_energy_error_J.abs().mean(), sup.window_energy_error_pct.abs().mean()))
