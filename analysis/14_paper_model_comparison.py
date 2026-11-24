"""Test the published arm model (Deniz et al.) on our arm-gesture data, in the paper's own framing
(net power relative to the pre-idle baseline of each trial), and quantify candidate improvements."""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1power as g
ROOT = sys.argv[1]; A = os.path.join(ROOT, "analysis")
pd.set_option("display.width", 250, "display.max_rows", 400, "display.max_columns", 60)
df0 = g.load(ROOT)
MODEL = sys.argv[2] if len(sys.argv) > 2 else "g1_power_model.json"      # pass g1_power_model_v2.json to decompose with the v2 coefficients
SUF = "_v2" if "v2" in MODEL else ""
M = json.load(open(os.path.join(ROOT, MODEL), encoding="utf-8"))
ARMT = g.TYPES[9:]                      # shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
LARM = list(range(15, 22)); RARM = list(range(22, 29))
# published coefficients (Table I / II), joint order = ARMT
PB = np.array([0.4184, 0.3563, 0.2799, 0.3942, 0.0, 0.3768, 0.0])
PC = np.array([0.0, 1.2767, 0.0, 0.0324, 0.0, 0.3464, 0.0])
PD = np.array([0.6467, 0.0, 0.2749, 0.0, 1.0282, 1.2004, 1.7848])
PE = {(0, 3): 0.9090, (0, 6): 0.4365, (1, 3): 1.1053, (1, 4): 0.1039, (1, 5): 0.1034, (1, 6): 0.5111, (2, 3): 0.4054,
      (2, 4): 0.0554, (2, 5): 0.0050, (2, 6): 0.8613, (3, 6): 0.0591, (4, 5): 0.3417, (5, 6): 0.2936}
PAIRS = [(i, j) for i in range(7) for j in range(i + 1, 7)]
GEST = ["hands up", "heart", "high five", "hug", "reject", "shake hand", "xray"]

# ------------------------------------------------------------------ phases per gesture segment
parts = []
for sid, gg in df0[df0.action.isin(GEST)].groupby("segid", sort=False):
    gg = gg.sort_values("t").reset_index(drop=True)
    if len(gg) < 150 or (gg.fsm == 1).any(): continue
    dq = gg[[f"dq{j}" for j in g.ARM]].abs().sum(1).values
    mv = np.where(dq > 1.0)[0]
    hold = gg.arm_holding.fillna(False).astype(bool).values
    if len(mv) == 0 or hold.sum() < 40: continue
    ms = mv[0]; h0 = np.where(hold)[0][0]; h1 = np.where(hold)[0][-1]
    idx = np.arange(len(gg))
    gg["is_pre"] = (idx >= 8) & (idx < ms - 5)
    gg["is_hold"] = hold & (dq < 0.6) & (idx >= h0 + 20) & (idx <= h1 - 3)
    gg["is_action"] = idx >= ms - 2
    if gg.is_pre.sum() < 15 or gg.is_hold.sum() < 20: continue
    parts.append(gg)
df = pd.concat(parts, ignore_index=True)
print("gesture segments used:", df.segid.nunique(), " by date:", df.groupby("date").segid.nunique().to_dict())
tau, dq, T = g.arrays(df)
bj = np.array([M["b"][t] for t in [n.replace("left_", "").replace("right_", "") for n in g.JN]])     # our per-joint b

# baselines per segment
t2 = tau ** 2
t2_pre = pd.DataFrame(np.where(df.is_pre.values[:, None], t2, np.nan)).groupby(df.segid.values).transform("mean").values
P_pre = df.P_bms.where(df.is_pre).groupby(df.segid).transform("mean").values
Pm_pre = df.P_mb.where(df.is_pre).groupby(df.segid).transform("mean").values
d2 = t2 - t2_pre
df["y_net"] = df.P_bms.values - P_pre
df["y_net_mb"] = df.P_mb.values - Pm_pre

print("\n== 1. static hold vs pre-idle: where does the net power come from? (our whole-body coefficients; mean over segments, W) ==")
H = df[df.is_hold]
hold_mean = lambda v: pd.Series(v[df.is_hold.values]).groupby(H.segid.values).mean()
tab = pd.DataFrame({"meas_BMS": hold_mean(df.y_net.values), "meas_mainboard": hold_mean(df.y_net_mb.values),
                    "arms": hold_mean((bj[g.ARM] * d2[:, g.ARM]).sum(1)), "waist": hold_mean((bj[g.WAIST] * d2[:, g.WAIST]).sum(1)),
                    "legs": hold_mean((bj[g.LEG] * d2[:, g.LEG]).sum(1)),
                    "paper_coefs_arms_only": hold_mean((np.r_[PB, PB] * d2[:, g.ARM]).sum(1))})
tab["ours_total"] = tab.arms + tab.waist + tab.legs
info = df.groupby("segid").agg(action=("action", "first"), date=("date", "first"), boot=("boot", "first"))
tab = tab.join(info)
ga = tab.groupby("action")[["meas_BMS", "meas_mainboard", "ours_total", "arms", "waist", "legs", "paper_coefs_arms_only"]].mean()
ga["arm_share_%"] = 100 * ga.arms / ga.meas_BMS
ga["n"] = tab.groupby("action").size()
print(ga.round(1).to_string())
lg = tab.groupby("action").legs.agg(["min", "max"])
print("legs term, min/max over segments by action:"); print(lg.round(1).T.to_string())
ga.assign(legs_min=lg["min"], legs_max=lg["max"]).round(2).to_csv(os.path.join(A, f"paper_cmp_hold_by_gesture{SUF}.csv"), encoding="utf-8-sig")
tab.round(2).to_csv(os.path.join(A, f"paper_cmp_hold_by_segment{SUF}.csv"), encoding="utf-8-sig")
if SUF:
    print("(v2 coefficients: only the decomposition above was requested; stopping here)"); sys.exit(0)
e_p = tab.paper_coefs_arms_only - tab.meas_BMS; e_o = tab.ours_total - tab.meas_BMS
print("hold-phase net power, %d segments: published arm coefficients  bias %.1f  MAE %.1f W | our whole-body (in-sample coefs)  bias %.1f  MAE %.1f W | mean measured %.1f W"
      % (len(tab), e_p.mean(), e_p.abs().mean(), e_o.mean(), e_o.abs().mean(), tab.meas_BMS.mean()))
print("measured hold-phase net power, mainboard minus BMS, by segment: mean %.2f  max |diff| %.2f W; by gesture max |diff| %.2f W"
      % ((tab.meas_mainboard - tab.meas_BMS).mean(), (tab.meas_mainboard - tab.meas_BMS).abs().max(), (ga.meas_mainboard - ga.meas_BMS).abs().max()))
# the baseline that the paper subtracts: how much does it vary, and can the absolute whole-body model predict it?
pm0 = tau * dq
P_abs = M["P0"] + (bj * t2).sum(1) + np.clip(pm0, 0, None).sum(1) + M["eta_regen"] * np.clip(pm0, None, 0).sum(1)
ip = df.is_pre.values
bp = pd.DataFrame({"meas": df.P_bms.values[ip], "model": P_abs[ip]}).groupby(df.segid.values[ip]).mean()
eb = bp.model - bp.meas
print("pre-idle baseline per segment (absolute BMS power): min %.1f  max %.1f  sd %.1f W | absolute whole-body model (final coefficients, in-sample): bias %.1f  MAE %.1f  max |err| %.1f W"
      % (bp.meas.min(), bp.meas.max(), bp.meas.std(), eb.mean(), eb.abs().mean(), eb.abs().max()))

# ------------------------------------------------------------------ regressors in the net-power framing
X = pd.DataFrame(index=df.index)
for t, idx in g.TYPE_IDX.items(): X[f"dcu_{t}"] = d2[:, idx].sum(1)
pm = tau * dq
X["mech_net"] = pm.sum(1); X["mech_pos"] = np.clip(pm, 0, None).sum(1); X["mech_neg"] = np.clip(pm, None, 0).sum(1)
aL, aR = np.abs(dq[:, LARM]), np.abs(dq[:, RARM])
for k, t in enumerate(ARMT):
    X[f"c_{t}"] = aL[:, k] + aR[:, k]; X[f"d_{t}"] = aL[:, k] ** 2 + aR[:, k] ** 2
for (i, j) in PAIRS: X[f"e_{i}{j}"] = aL[:, i] * aL[:, j] + aR[:, i] * aR[:, j]
# published model, instantaneous net power
paper = (X[[f"dcu_{t}" for t in ARMT]].values @ PB + X.mech_net.values + X[[f"c_{t}" for t in ARMT]].values @ PC
         + X[[f"d_{t}" for t in ARMT]].values @ PD + sum(v * X[f"e_{i}{j}"].values for (i, j), v in PE.items()))
X["paper_pred"] = paper
meta, Xf = g.bms_align(df, X)
y = meta.y_net.values
act = meta.is_action.values; hold = meta.is_hold.values

def report(label, pred, k):
    e = pred - y
    seg_h = pd.Series(e[hold]).groupby(meta.segid.values[hold]).mean()
    seg_a = pd.Series(e[act]).groupby(meta.segid.values[act]).mean()
    mov = act & ~hold
    return dict(model=label, k=k, rmse_all=np.sqrt((e ** 2).mean()), rmse_moving=np.sqrt((e[mov] ** 2).mean()), hold_bias=seg_h.mean(), hold_mae=seg_h.abs().mean(),
                hold_max=seg_h.abs().max(), traj_mae=seg_a.abs().mean(), traj_max=seg_a.abs().max())

ARM_B = [f"dcu_{t}" for t in ARMT]; ALL_B = [f"dcu_{t}" for t in g.TYPES]
ARM_CD = [f"c_{t}" for t in ARMT] + [f"d_{t}" for t in ARMT]; ARM_E = [f"e_{i}{j}" for (i, j) in PAIRS]
MODELS = {
 "A1 arm only: b,c,d (a=1)": (ARM_B + ARM_CD + ["mech_net"], dict(fixed={"mech_net": 1.0})),
 "A2 arm only: b,c,d,e_ij (paper structure)": (ARM_B + ARM_CD + ARM_E + ["mech_net"], dict(fixed={"mech_net": 1.0})),
 "W1 whole-body b (a=1), no c,d": (ALL_B + ["mech_net"], dict(fixed={"mech_net": 1.0})),
 "W2 whole-body b, braking coef free": (ALL_B + ["mech_pos", "mech_neg"], dict(fixed={"mech_pos": 1.0}, upper={"mech_neg": 1.0})),
 "W3 W2 + arm c,d": (ALL_B + ARM_CD + ["mech_pos", "mech_neg"], dict(fixed={"mech_pos": 1.0}, upper={"mech_neg": 1.0})),
 "W4 W2 + arm c,d,e_ij": (ALL_B + ARM_CD + ARM_E + ["mech_pos", "mech_neg"], dict(fixed={"mech_pos": 1.0}, upper={"mech_neg": 1.0})),
}
allrows = []
for cvname, groups in [("leave-one-boot-out", meta.boot.values), ("leave-one-gesture-out", meta.action.values)]:
    rows = [report("P0 published coefficients, no fitting", Xf.paper_pred.values, 0)]
    for name, (cols, kw) in MODELS.items():
        p, C = g.lobo_cv(meta, Xf[cols], y, groups, **kw)
        rows.append(report(name, p, len(cols) - len(kw.get("fixed", {}))))
    print(f"\n== 2. net power (BMS, relative to pre-idle), {cvname}; n={len(y)} BMS samples, {meta.segid.nunique()} segments ==")
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    allrows += [dict(split=cvname, **r) for r in rows]
pd.DataFrame(allrows).round(3).to_csv(os.path.join(A, "paper_cmp_validation.csv"), index=False, encoding="utf-8-sig")

print("\n== 2b. leave-one-gesture-out: hold-phase error of the held-out gesture (segment means, W) ==")
for name in ["A1 arm only: b,c,d (a=1)", "W2 whole-body b, braking coef free"]:
    cols, kw = MODELS[name]
    p, _ = g.lobo_cv(meta, Xf[cols], y, meta.action.values, **kw)
    sh = pd.Series((p - y)[hold]).groupby(meta.segid.values[hold]).mean()
    a = pd.Series(meta.action.values[hold]).groupby(meta.segid.values[hold]).first()
    r = sh.groupby(a).agg(bias="mean", mae=lambda s: s.abs().mean(), max=lambda s: s.abs().max())
    print(name); print(r.round(1).T.to_string())
    r.round(2).rename_axis("action").reset_index().assign(model=name).to_csv(
        os.path.join(A, "paper_cmp_leave_gesture_out.csv"), index=False, encoding="utf-8-sig", mode="w" if name.startswith("A1") else "a", header=name.startswith("A1"))

print("\n== 3. arm copper coefficients: arm-only fit vs whole-body fit vs published (all gesture data) ==")
cA = g.fit_bounded(Xf[MODELS["A1 arm only: b,c,d (a=1)"][0]], y, fixed={"mech_net": 1.0})
cW = g.fit_bounded(Xf[MODELS["W2 whole-body b, braking coef free"][0]], y, fixed={"mech_pos": 1.0}, upper={"mech_neg": 1.0})
ctab = pd.DataFrame({"published": PB, "arm_only_fit": [cA[f"dcu_{t}"] for t in ARMT], "whole_body_fit_net": [cW[f"dcu_{t}"] for t in ARMT],
                     "whole_body_fit_absolute": [M["b"][t] for t in ARMT]}, index=ARMT)
print(ctab.round(3).to_string())
ctab.round(4).to_csv(os.path.join(A, "paper_cmp_arm_coefs.csv"), encoding="utf-8-sig")
print("arm-only fit c (W s/rad):", {t: round(cA[f'c_{t}'], 2) for t in ARMT}); print("arm-only fit d:", {t: round(cA[f'd_{t}'], 2) for t in ARMT})
print("braking coefficient in W2:", round(cW["mech_neg"], 3))

print("\n== 4. trajectory-level aggregation (one row per trial, as in the paper) vs per-sample fit with the sensor response ==")
agg = Xf[act].groupby(meta.segid.values[act]).mean(); ya = pd.Series(y[act]).groupby(meta.segid.values[act]).mean()
ga_ = pd.Series(meta.boot.values[act]).groupby(meta.segid.values[act]).first().values
gact = pd.Series(meta.action.values[act]).groupby(meta.segid.values[act]).first().values
ma = pd.DataFrame({"segid": agg.index, "boot": ga_}).set_index(agg.index)
for name in ["A1 arm only: b,c,d (a=1)", "A2 arm only: b,c,d,e_ij (paper structure)", "W2 whole-body b, braking coef free"]:
    cols, kw = MODELS[name]
    free = [c for c in cols if c not in kw.get("fixed", {})]
    sv = np.linalg.svd(agg[free].values / np.sqrt((agg[free].values ** 2).mean(0) + 1e-12), compute_uv=False)
    out = []
    for cvname, grp in [("boot", ga_), ("gesture", gact)]:
        p_tr, _ = g.lobo_cv(ma, agg[cols], ya.values, grp, **kw)                       # fitted on trial means
        grp_s = meta.boot.values if cvname == "boot" else meta.action.values
        p_s, _ = g.lobo_cv(meta, Xf[cols], y, grp_s, **kw)                             # fitted per sample
        p_s_agg = pd.Series(p_s[act]).groupby(meta.segid.values[act]).mean().reindex(agg.index).values
        out.append(f"leave-{cvname}-out trial-mean MAE: fitted on trial means {np.abs(p_tr - ya.values).mean():.2f} W, fitted per sample {np.abs(p_s_agg - ya.values).mean():.2f} W")
    print(f"{name}: rows {len(agg)}, free params {len(free)}, effective rank (sv > 1e-3*max) {(sv > 1e-3 * sv[0]).sum()}, cond {sv[0] / max(sv[-1], 1e-12):.1e}\n   " + "\n   ".join(out))

print("\n== 5. same comparison with the mainboard reading as target (10 Hz, per frame, no sensor filter), leave-one-boot-out ==")
mm = df.P_mb.notna().values & df.y_net_mb.notna().values
ym = df.y_net_mb.values[mm]; Xm = X[mm]; dm = df[mm]
rows = []
def rep_mb(label, p):
    e = p - ym; h = dm.is_hold.values; a = dm.is_action.values
    sh = pd.Series(e[h]).groupby(dm.segid.values[h]).mean()
    return dict(model=label, rmse_all=np.sqrt((e ** 2).mean()), rmse_moving=np.sqrt((e[a & ~h] ** 2).mean()), hold_bias=sh.mean(), hold_mae=sh.abs().mean(), hold_max=sh.abs().max())
rows.append(rep_mb("P0 published coefficients, no fitting", Xm.paper_pred.values))
for name in ["A1 arm only: b,c,d (a=1)", "A2 arm only: b,c,d,e_ij (paper structure)", "W2 whole-body b, braking coef free", "W3 W2 + arm c,d"]:
    cols, kw = MODELS[name]
    p, _ = g.lobo_cv(dm, Xm[cols], ym, dm.boot.values, **kw); rows.append(rep_mb(name, p))
print(pd.DataFrame(rows).round(2).to_string(index=False))
