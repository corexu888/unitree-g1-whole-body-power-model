"""Model v2: copper coefficients tied by actuator model (g.ACT_GROUPS), fitted on the BMS target as in v1.
   Writes g1_power_model_v2.json and the v1-vs-v2 validation tables; also tests two add-ons on held-out data:
   an uncertainty estimate for a segment mean, and a one-off offset calibration at the start of a session."""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1power as g
ROOT = sys.argv[1]; A = os.path.join(ROOT, "analysis")
pd.set_option("display.width", 250, "display.max_rows", 400, "display.max_columns", 60)
df = g.load(ROOT)
KW = dict(fixed={"mech_pos": 1.0}, upper={"mech_neg": 1.0})
X16 = g.features(df, dict(cu="type", mech="split")); X16.insert(0, "const", 1.0)
X16T = g.features(df, g.V2_SPEC); X16T.insert(0, "const", 1.0)       # copper terms scaled by 1 + alpha_Cu * (T - T at segment start)
meta_all, Xcat = g.bms_align(df, pd.concat([X16, X16T.add_suffix("@T")], axis=1)); k = (meta_all.fsm != 1).values
Xf16_all = Xcat[X16.columns]; Xf16T_all = Xcat[[c + "@T" for c in X16.columns]].rename(columns=lambda c: c[:-2])
meta, Xf16 = meta_all[k], Xf16_all[k]; y = meta.P_bms.values
Xtied = g.tie_types(Xf16)                                # tied coefficients only
Xf = g.tie_types(Xf16T_all[k])                           # v2 regressors: tied coefficients + temperature-rise factor
V = {"v1": Xf16, "tied": Xtied, "v2": Xf}
hold = meta.arm_holding.fillna(False).astype(bool).values
def hold_slope(p, mask):
    """mean slope of (measured - model) against time inside arm holds, W per 10 s"""
    out = [np.polyfit(x.t - x.t.iloc[0], x.r, 1)[0] * 10 for s, x in pd.DataFrame({"t": meta.t.values[mask], "r": (y - p)[mask]}).groupby(meta.segid.values[mask]) if len(x) >= 12]
    return float(np.mean(out)), len(out)
segid = meta.segid.values
seg_info = meta.groupby("segid").agg(action=("action", "first"), boot=("boot", "first"), date=("date", "first"), robot=("robot", "first"), t0=("t", "min"))

def seg_mean(v): return pd.Series(v).groupby(segid).mean()
ybar = seg_mean(y)

# ------------------------------------------------------------------ 1. final coefficients and bootstrap
coef = g.fit_bounded(Xf, y, **KW)
rng = np.random.default_rng(1)
segs = np.array(sorted(set(segid))); rows_of = {s: np.where(segid == s)[0] for s in segs}
def boot_fit(Xsrc, yy, pool, n, rows=rows_of):
    out = []
    for r in range(n):
        idx = np.concatenate([rows[s] for s in rng.choice(pool, len(pool))])
        out.append(g.fit_bounded(Xsrc.iloc[idx], yy[idx], **KW).values)
    return np.array(out)
BS = pd.DataFrame(boot_fit(Xf.reset_index(drop=True), y, segs, 300), columns=Xf.columns)
ci = pd.DataFrame({"coef": coef, "lo": BS.quantile(.025), "hi": BS.quantile(.975)})
def conf(name):
    if name == "mech_pos": return "fixed (energy conservation)"
    c, lo, hi = ci.loc[name, "coef"], ci.loc[name, "lo"], ci.loc[name, "hi"]
    if name == "const": return "high"
    if c <= 0 or lo <= 0.02 * max(c, 1e-9): return "not identified"
    rel = (hi - lo) / 2 / c
    return "high" if rel <= 0.25 else ("medium" if rel <= 0.6 else "low")
ci["confidence"] = [conf(n) for n in ci.index]
print("== v2 coefficients (all data), 95% interval from 300 resamples of segments =="); print(ci.round(4).to_string())
ci.round(5).to_csv(os.path.join(A, "v2_coef_table.csv"), encoding="utf-8-sig")

# ------------------------------------------------------------------ 2. held-out validation, v1 and v2 side by side
def split_eval(Xsrc, train, test):
    c = g.fit_bounded(Xsrc[train], y[train], **KW); p = Xsrc.values @ c.values
    e = pd.Series((p - y)[test]).groupby(segid[test]).mean()
    return e
acts = meta.action.values; boots = meta.boot.values
gest = ~meta.action.isin(g.LOCO + ["stand still", "squat"]).values; loco = meta.action.isin(g.LOCO).values; isA = (meta.robot == "A").values
rows = []; segerr = {}
for ver, Xsrc in V.items():
    p, C = g.lobo_cv(meta, Xsrc, y, boots, **KW); e = seg_mean(p - y); segerr[ver] = e
    bb = pd.Series(p - y).groupby(boots).mean()
    s_rej, n_rej = hold_slope(p, hold & (acts == "reject")); s_oth, n_oth = hold_slope(p, hold & (acts != "reject"))
    rows.append(dict(model=ver, test="held-out power-on session (9 folds)", n_seg=len(e), mae=e.abs().mean(), median=e.abs().median(), p90=e.abs().quantile(.9), max=e.abs().max(), bias=e.mean(),
                     extra="session bias %.1f … %+.1f; drift inside holds: reject %+.1f W/10 s (%d holds), other gestures %+.1f W/10 s (%d holds)" % (bb.min(), bb.max(), s_rej, n_rej, s_oth, n_oth)))
    pa, _ = g.lobo_cv(meta, Xsrc, y, acts, **KW); ea = seg_mean(pa - y); by = ea.groupby(seg_info.action.reindex(ea.index).values).mean()
    rows.append(dict(model=ver, test="held-out action class (12 folds)", n_seg=len(ea), mae=ea.abs().mean(), median=ea.abs().median(), p90=ea.abs().quantile(.9), max=ea.abs().max(), bias=ea.mean(),
                     extra="worst class: %s %+.1f" % (by.abs().idxmax(), by[by.abs().idxmax()])))
    for label, tr, te in [("fit on robot B, test on robot A", ~isA, isA), ("fit without locomotion, test on locomotion", ~loco, loco),
                          ("fit without arm gestures, test on arm gestures", ~gest, gest), ("fit on arm gestures only, test on the rest", gest, ~gest)]:
        e2 = split_eval(Xsrc, tr, te)
        rows.append(dict(model=ver, test=label, n_seg=len(e2), mae=e2.abs().mean(), median=e2.abs().median(), p90=e2.abs().quantile(.9), max=e2.abs().max(), bias=e2.mean(), extra=""))
    for d in sorted(meta.date.unique()):
        te = (meta.date == d).values; e2 = split_eval(Xsrc, ~te, te)
        rows.append(dict(model=ver, test=f"held-out date {d}", n_seg=len(e2), mae=e2.abs().mean(), median=e2.abs().median(), p90=e2.abs().quantile(.9), max=e2.abs().max(), bias=e2.mean(), extra=""))
R = pd.DataFrame(rows)
piv = R.pivot(index="test", columns="model", values=["mae", "median", "p90", "max", "bias"]).round(2)
print("\n== held-out segment-mean errors (W): v1, tied coefficients only, v2 (tied + temperature rise) =="); print(piv.to_string())
for r_ in R[R.extra != ""].itertuples(): print(f"{r_.model:5s} | {r_.test} | {r_.extra}")
R.round(3).to_csv(os.path.join(A, "v2_holdout_splits.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame({"meas": ybar, "err_v1": segerr["v1"], "err_tied": segerr["tied"], "err_v2": segerr["v2"]}).join(seg_info).round(3).to_csv(os.path.join(A, "v2_segment_errors_cv.csv"), encoding="utf-8-sig")

# damping (zero-torque) frames, never used for fitting
dmp = (meta_all.fsm == 1).values
if dmp.any():
    pd_ = g.tie_types(Xf16T_all[dmp]).values @ coef.values
    print("\ndamping state (not fitted): measured vs v2 by robot (W):")
    print(pd.DataFrame({"meas": meta_all.P_bms.values[dmp], "v2": pd_, "robot": meta_all.robot.values[dmp]}).groupby("robot").agg(n=("meas", "size"), meas=("meas", "mean"), v2=("v2", "mean")).round(1).to_string())

# ------------------------------------------------------------------ 3. uncertainty of a segment mean, tested on held-out folds
# total sd^2 = parameter part (bootstrap inside the training fold) + session-to-session part + segment-to-segment part
Xr = Xf.reset_index(drop=True)
def fold_unc(groups, nrep=100):
    pred = pd.Series(index=segs, dtype=float); psd = pd.Series(index=segs, dtype=float)
    for b in sorted(pd.unique(groups)):
        te = groups == b; tr_segs = np.array(sorted(set(segid[~te]))); te_segs = sorted(set(segid[te]))
        c = g.fit_bounded(Xf[~te], y[~te], **KW).values
        Bf = boot_fit(Xr, y, tr_segs, nrep)
        for s in te_segs:
            xb = Xr.iloc[rows_of[s]].mean().values
            pred[s] = xb @ c; psd[s] = (Bf @ xb).std()
    return pred, psd
p_lobo, sd_lobo = fold_unc(boots)
e_lobo = p_lobo - ybar
bias_b = e_lobo.groupby(seg_info.boot.reindex(e_lobo.index).values).transform("mean")
s_boot = float(e_lobo.groupby(seg_info.boot.reindex(e_lobo.index).values).mean().std(ddof=1)); s_seg = float((e_lobo - bias_b).std(ddof=1))
tot = np.sqrt(sd_lobo ** 2 + s_boot ** 2 + s_seg ** 2)
print("\n== uncertainty of a segment mean (v2) ==")
print("session-to-session sd %.2f W, segment-to-segment sd %.2f W, parameter sd per segment: median %.2f, max %.2f W" % (s_boot, s_seg, sd_lobo.median(), sd_lobo.max()))
print("held-out session: |error| <= 1.96 sd in %.0f%% of %d segments; <= 1 sd in %.0f%%" % (100 * (e_lobo.abs() <= 1.96 * tot).mean(), len(tot), 100 * (e_lobo.abs() <= tot).mean()))
p_lao, sd_lao = fold_unc(acts)
e_lao = p_lao - ybar; tot_a = np.sqrt(sd_lao ** 2 + s_boot ** 2 + s_seg ** 2)
ua = pd.DataFrame({"err": e_lao, "param_sd": sd_lao, "total_sd": tot_a, "action": seg_info.action.reindex(e_lao.index).values})
print("held-out action class: |error| <= 1.96 sd in %.0f%% of segments" % (100 * (ua.err.abs() <= 1.96 * ua.total_sd).mean()))
print(ua.groupby("action").agg(n=("err", "size"), bias=("err", "mean"), max_abs=("err", lambda s: s.abs().max()), param_sd=("param_sd", "mean"), total_sd=("total_sd", "mean"),
                               covered=("err", lambda s: float((s.abs() <= 1.96 * ua.total_sd[s.index]).mean()))).round(2).to_string())
ua.round(3).to_csv(os.path.join(A, "v2_uncertainty_leave_action_out.csv"), encoding="utf-8-sig")

# ------------------------------------------------------------------ 4. one-off offset calibration at the start of a session (held-out session)
print("\n== session offset calibration (v2, held-out session): offset = measured - model over the first part of the session ==")
p, C = g.lobo_cv(meta, Xf, y, boots, **KW); r = pd.Series(y - p, index=meta.index)
res = []
for b in sorted(pd.unique(boots)):
    mb = meta[boots == b].sort_values("t"); rb = r[mb.index]
    order = mb.groupby("segid").t.min().sort_values().index.tolist()
    if len(order) < 3: continue
    for label, cal in [("first 10 s", mb.index[(mb.t - mb.t.iloc[0]) <= 10.0]), ("first segment", mb.index[mb.segid == order[0]])]:
        o = rb[cal].mean(); rest = mb.index[mb.segid != order[0]]
        e0 = rb[rest].groupby(mb.segid[rest].values).mean(); e1 = e0 - o
        res.append(dict(boot=b, calib=label, offset=o, n_rest=len(e0), mae_before=e0.abs().mean(), mae_after=e1.abs().mean(), max_before=e0.abs().max(), max_after=e1.abs().max()))
RC = pd.DataFrame(res); print(RC.round(2).to_string(index=False))
for label, x in RC.groupby("calib"):
    wgt = x.n_rest / x.n_rest.sum()
    print(f"{label}: segment MAE over the remaining segments {float((x.mae_before * wgt).sum()):.2f} -> {float((x.mae_after * wgt).sum()):.2f} W; worst segment {x.max_before.max():.1f} -> {x.max_after.max():.1f} W")
RC.round(3).to_csv(os.path.join(A, "v2_session_calibration.csv"), index=False, encoding="utf-8-sig")

# ------------------------------------------------------------------ 5. write the model file
tau, dq, T = g.arrays(df); m = (df.fsm != 1).values
env = {t: dict(tau_abs_p99=float(np.percentile(np.abs(tau[m][:, idx]), 99)), tau_abs_max=float(np.abs(tau[m][:, idx]).max()),
               dq_abs_p99=float(np.percentile(np.abs(dq[m][:, idx]), 99)), dq_abs_max=float(np.abs(dq[m][:, idx]).max())) for t, idx in g.TYPE_IDX.items()}
grp_of = {t: gname for gname, types in g.ACT_GROUPS.items() for t in types}
try: mbfir = json.load(open(os.path.join(A, "v2_mb_fir.json")))
except Exception: mbfir = None
model = dict(
    name="G1 whole-robot electrical power model (BMS pack power), v2", version="v2",
    equation="P[W] = P0 + sum_groups b_group * sum_{j in group} tau_j^2 * (1 + alpha_cu*(T_j - T_j,start)) + sum_j max(tau_j*dq_j,0) + eta_regen * sum_j min(tau_j*dq_j,0)",
    change_from_v1="(1) joint types driven by the same actuator model share one copper coefficient (8 instead of 16); (2) copper loss follows the rise of the reported motor "
                   "temperature since the start of the segment with the temperature coefficient of copper (not fitted). Same data, target and fitting method as v1.",
    alpha_cu=g.ALPHA_CU, temp_reference="each motor's reported temperature averaged over the first 1.5 s of the segment (robot standing, arms down)",
    units=dict(tau="N*m (lowstate tau_est)", dq="rad/s", P="W, battery pack V*|I| as reported by the BMS"),
    joint_order=g.JN, P0=float(coef["const"]), eta_regen=float(coef["mech_neg"]), mech_pos_coef=1.0,
    groups=g.ACT_GROUPS, b_group={gn: float(coef[f"cu_{gn}"]) for gn in g.ACT_GROUPS},
    b_group_ci95={gn: [float(ci.loc[f"cu_{gn}", "lo"]), float(ci.loc[f"cu_{gn}", "hi"])] for gn in g.ACT_GROUPS},
    b={t: float(coef[f"cu_{grp_of[t]}"]) for t in g.TYPES},
    b_ci95={t: [float(ci.loc[f"cu_{grp_of[t]}", "lo"]), float(ci.loc[f"cu_{grp_of[t]}", "hi"])] for t in g.TYPES},
    P0_ci95=[float(ci.loc["const", "lo"]), float(ci.loc["const", "hi"])], eta_regen_ci95=[float(ci.loc["mech_neg", "lo"]), float(ci.loc["mech_neg", "hi"])],
    confidence={**{"P0": "high", "mech_neg": ci.loc["mech_neg", "confidence"]}, **{f"b_{t}": ci.loc[f"cu_{grp_of[t]}", "confidence"] for t in g.TYPES}},
    bms_response_fir_10hz=[float(v) for v in g.BMS_FIR], mainboard_response=mbfir,
    uncertainty=dict(note="sd of a segment-mean prediction = sqrt(parameter part + session_sd^2 + segment_sd^2); parameter part from the bootstrap coefficient sets below",
                     session_sd=s_boot, segment_sd=s_seg, bootstrap_columns=["P0"] + [f"b_{gn}" for gn in g.ACT_GROUPS] + ["eta_regen"],
                     bootstrap_samples=np.round(BS[["const"] + [f"cu_{gn}" for gn in g.ACT_GROUPS] + ["mech_neg"]].values, 5).tolist()),
    fitted_on=dict(rows=int(len(y)), segments=int(meta.segid.nunique()), boots=int(meta.boot.nunique()), robots=["A", "B"],
                   dates=sorted(meta.date.unique().tolist()), excluded="damping (fsm=1) frames; first 0.8 s of each segment"),
    envelope=dict(per_joint_type=env, pack_voltage_V=[float(df.volt_v.min()), float(df.volt_v.max())], soc_percent=[float(df.soc.min()), float(df.soc.max())],
                  motor_temp_C=[float(T.min()), float(T.max())], power_W_bms=[float(df.P_bms[m].min()), float(df.P_bms[m].max())],
                  conditions="standing / arm gestures / 0.5 m/s commanded walk, step in place, turning; flat floor; slack safety rope; ~28 C ambient; no payload"))
json.dump(model, open(os.path.join(ROOT, "g1_power_model_v2.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("\nwrote g1_power_model_v2.json: P0 %.2f  eta_regen %.3f" % (model["P0"], model["eta_regen"]))
