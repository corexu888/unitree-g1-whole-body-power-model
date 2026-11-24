"""Final fit, validity envelope, figures and tables for the report."""
import sys, os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1power as g
ROOT = sys.argv[1]; A = os.path.join(ROOT, "analysis"); FIG = os.path.join(A, "fig"); os.makedirs(FIG, exist_ok=True)
df = g.load(ROOT)
SPEC = dict(cu="type", mech="split"); KW = dict(fixed={"mech_pos": 1.0}, upper={"mech_neg": 1.0})
X = g.features(df, SPEC); X.insert(0, "const", 1.0)
meta_all, Xf_all = g.bms_align(df, X); k = (meta_all.fsm != 1).values
meta, Xf = meta_all[k], Xf_all[k]; y = meta.P_bms.values
coef = g.fit_bounded(Xf, y, **KW)
pred_cv, C = g.lobo_cv(meta, Xf, y, meta.boot.values, **KW)
ct = pd.read_csv(os.path.join(A, "coef_table.csv"), index_col=0)
tau, dq, T = g.arrays(df)

# ---- confidence class per coefficient
def conf(name):
    if name in ("const",): return "high"
    if name == "mech_pos": return "fixed (energy conservation)"
    lo, hi, c = ct.loc[name, "bs_2.5%"], ct.loc[name, "bs_97.5%"], ct.loc[name, "coef"]
    if c <= 0 or lo <= 0.02 * max(c, 1e-9): return "not identified"
    rel = (hi - lo) / 2 / c
    return "high" if rel <= 0.25 else ("medium" if rel <= 0.6 else "low")
conf_map = {n: conf(n) for n in coef.index}

# ---- validity envelope (fitting data, damping excluded)
m = (df.fsm != 1).values
env = {}
for t, idx in g.TYPE_IDX.items():
    env[t] = dict(tau_abs_p99=float(np.percentile(np.abs(tau[m][:, idx]), 99)), tau_abs_max=float(np.abs(tau[m][:, idx]).max()),
                  dq_abs_p99=float(np.percentile(np.abs(dq[m][:, idx]), 99)), dq_abs_max=float(np.abs(dq[m][:, idx]).max()))
model = dict(
    name="G1 whole-robot electrical power model (BMS pack power), v1",
    equation="P[W] = P0 + sum_types b_type * sum_{j in type} tau_j^2 + sum_j max(tau_j*dq_j,0) + eta_regen * sum_j min(tau_j*dq_j,0)",
    units=dict(tau="N*m (lowstate tau_est)", dq="rad/s", P="W, battery pack V*|I| as reported by the BMS"),
    joint_order=g.JN, P0=float(coef["const"]), eta_regen=float(coef["mech_neg"]), mech_pos_coef=1.0,
    b={t: float(coef[f"cu_{t}"]) for t in g.TYPES},
    b_ci95={t: [float(ct.loc[f"cu_{t}", "bs_2.5%"]), float(ct.loc[f"cu_{t}", "bs_97.5%"])] for t in g.TYPES},
    P0_ci95=[float(ct.loc["const", "bs_2.5%"]), float(ct.loc["const", "bs_97.5%"])],
    eta_regen_ci95=[float(ct.loc["mech_neg", "bs_2.5%"]), float(ct.loc["mech_neg", "bs_97.5%"])],
    confidence={("P0" if n == "const" else n.replace("cu_", "b_")): c for n, c in conf_map.items()},
    bms_response_fir_10hz=[float(v) for v in g.BMS_FIR],
    fitted_on=dict(rows=int(len(y)), segments=int(meta.segid.nunique()), boots=int(meta.boot.nunique()), robots=["A", "B"],
                   dates=sorted(meta.date.unique().tolist()), excluded="damping (fsm=1) frames; first 0.8 s of each segment"),
    envelope=dict(per_joint_type=env, pack_voltage_V=[float(df.volt_v.min()), float(df.volt_v.max())], soc_percent=[float(df.soc.min()), float(df.soc.max())],
                  motor_temp_C=[float(T.min()), float(T.max())], power_W_bms=[float(df.P_bms[m].min()), float(df.P_bms[m].max())],
                  conditions="standing / arm gestures / 0.5 m/s commanded walk, step in place, turning; flat floor; slack safety rope; ~28 C ambient; no payload"))
json.dump(model, open(os.path.join(ROOT, "g1_power_model.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("P0 %.2f  eta_regen %.3f" % (model["P0"], model["eta_regen"]))
print(pd.DataFrame({"b": model["b"], "conf": {t: conf_map[f"cu_{t}"] for t in g.TYPES}, **{k2: {t: env[t][k2] for t in g.TYPES} for k2 in ["tau_abs_p99", "tau_abs_max", "dq_abs_p99"]}}).round(4).to_string())

# ---- per-frame LOBO predictions (instantaneous and BMS-filtered) for plots
inst = np.zeros(len(df))
for b in sorted(df.boot.unique()):
    mm = (df.boot == b).values
    inst[mm] = X[mm].values @ C[b].reindex(X.columns).values
df["pred_inst"] = inst
K = len(g.BMS_FIR) - 1
df["pred_bms"] = df.groupby("segid").pred_inst.transform(lambda s: np.convolve(s.values, g.BMS_FIR, mode="full")[:len(s)])
df.loc[df.groupby("segid").cumcount() < K, "pred_bms"] = np.nan

# ---- Fig 1: segment means, measured vs predicted (LOBO-CV)
seg = pd.read_csv(os.path.join(A, "segment_errors_cv.csv"))
fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
cols = {"2026-08-04": "tab:red", "2026-11-06": "tab:orange", "2026-11-11": "tab:green", "2026-11-13": "tab:blue", "2026-11-18": "tab:purple"}
for d, gg in seg.groupby("date"):
    lo = gg.action.isin(g.LOCO)
    ax[0].scatter(gg.meas[~lo], gg.pred[~lo], c=cols[d], s=26, label=f"{d} ({'A' if d < '2026-09' else 'B'}) arm/stand", alpha=.8)
    if lo.any(): ax[0].scatter(gg.meas[lo], gg.pred[lo], c=cols[d], s=40, marker="^", label=f"{d} locomotion", alpha=.8)
lim = [110, 195]; ax[0].plot(lim, lim, "k-", lw=.8); ax[0].plot(lim, [lim[0] + 5, lim[1] + 5], "k:", lw=.6); ax[0].plot(lim, [lim[0] - 5, lim[1] - 5], "k:", lw=.6)
ax[0].set_xlabel("measured segment-mean BMS power (W)"); ax[0].set_ylabel("predicted, session held out (W)"); ax[0].legend(fontsize=7); ax[0].set_title("Segment means, leave-one-boot-out (dotted: +/-5 W)")
ax[1].hist(seg.e, bins=np.arange(-8, 8.5, 1), color="gray"); ax[1].set_xlabel("predicted - measured (W), segment mean"); ax[1].set_ylabel("segments")
ax[1].set_title("111 segments: median |err| %.1f W, 90%% < %.1f W, max %.1f W" % (seg.e.abs().median(), seg.e.abs().quantile(.9), seg.e.abs().max()))
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig1_segment_means_cv.png"), dpi=110); plt.close()

# ---- Fig 2: time series examples
picks = [("2026-11-18", "Arm\\Heart - Run 1"), ("2026-11-18", "Arm\\Reject - Run 1"), ("2026-11-13", "stand still - Run 1 - exec157031804"),
         ("2026-11-18", "Arm\\Stand Still - Run 1"), ("2026-11-18", "Locomotion\\Walk forward (Run 1)"), ("2026-11-06", "Walk forward - Run 1"),
         ("2026-08-04", "squat"), ("2026-08-04", "hug")]
fig, axes = plt.subplots(4, 2, figsize=(15, 13)); axes = axes.ravel()
for ax, (d, s) in zip(axes, picks):
    gg = df[(df.date == d) & (df.seg == s)]; t = gg.t - gg.t.iloc[0]
    ax.plot(t, gg.P_bms, "k-", lw=1.6, label="measured (BMS)")
    ax.plot(t, gg.pred_bms, "r-", lw=1.3, label="model, session held out (through BMS response)")
    ax.plot(t, gg.pred_inst, color="tab:blue", lw=.6, alpha=.6, label="model, instantaneous (10 Hz)")
    ax.axhline(coef["const"], color="gray", ls=":", lw=.8)
    ax.set_title(f"{d} robot {gg.robot.iloc[0]} | {s}", fontsize=9); ax.set_ylabel("W"); ax.set_xlabel("s")
axes[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig2_time_series.png"), dpi=100); plt.close()

# ---- Fig 3: per-boot bias and stand-still check
R = pd.read_pickle(os.path.join(A, "resid_cv.pkl"))
bb = R.groupby(["boot", "date", "robot"]).apply(lambda x: pd.Series({"bias": x.e.mean(), "off": (x.P_bms - x.P_mb).mean(), "n": len(x)})).reset_index()
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
lab = [f"{r.date[5:]}\n#{int(r.boot)} {r.robot}" for r in bb.itertuples()]
ax[0].bar(range(len(bb)), bb.bias, color=[cols[d] for d in bb.date]); ax[0].set_xticks(range(len(bb))); ax[0].set_xticklabels(lab, fontsize=8)
ax[0].axhline(0, color="k", lw=.6); ax[0].set_ylabel("mean (pred - meas), W"); ax[0].set_title("Bias per power-on session when that session is held out")
ss = seg[seg.action == "stand still"]
ax[1].scatter(ss.meas, ss.pred, c=[cols[d] for d in ss.date], s=40); ax[1].plot([112, 165], [112, 165], "k-", lw=.8)
ax[1].set_xlabel("measured standing power (W)"); ax[1].set_ylabel("predicted (W)"); ax[1].set_title("Standing still: 118-161 W spread is explained by leg torques")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig3_session_bias_standing.png"), dpi=110); plt.close()

# ---- tables for the report
seg["absdq"] = 0
by_act = seg.groupby("action").agg(n=("e", "size"), meas_min=("meas", "min"), meas_max=("meas", "max"), bias=("e", "mean"), mae=("e", lambda x: x.abs().mean()), worst=("e", lambda x: x.abs().max())).round(1)
print("\nper-action segment-level LOBO-CV errors:"); print(by_act.to_string())
by_act.to_csv(os.path.join(A, "errors_by_action_cv.csv"), encoding="utf-8-sig")
# typical decomposition of power for representative states (all-data coefficients, BMS-aligned rows)
comp = pd.DataFrame({c: coef[c] * Xf[c] for c in Xf.columns})
grp = {"P0": ["const"], "legs tau2": [f"cu_{t}" for t in g.TYPES[:6]], "waist tau2": [f"cu_{t}" for t in g.TYPES[6:9]], "arms tau2": [f"cu_{t}" for t in g.TYPES[9:]], "mech": ["mech_pos", "mech_neg"]}
cg = pd.DataFrame({k2: comp[v].sum(1) for k2, v in grp.items()}); cg["meas"] = y; cg["action"] = meta.action.values
print("\nmean decomposition by action (W):"); print(cg.groupby("action").mean().round(1).to_string())
cg.groupby("action").mean().round(1).to_csv(os.path.join(A, "decomposition_by_action.csv"), encoding="utf-8-sig")
