"""Diagnostic time-series plots for representative segments."""
import sys, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = sys.argv[1]; A = os.path.join(ROOT, "analysis"); FIG = os.path.join(A, "fig"); os.makedirs(FIG, exist_ok=True)
df = pd.read_pickle(os.path.join(A, "frames.pkl")).sort_values("t").reset_index(drop=True)
df["P_bms"] = df.volt_v * df.cur_a.abs(); df["P_mb"] = df.mb_value0 * df.mb_value2
LEG = list(range(0, 12)); WAIST = [12, 13, 14]; ARM = list(range(15, 29))
tau = df[[f"tau{j}" for j in range(29)]].values; dq = df[[f"dq{j}" for j in range(29)]].values
df["tau2_leg"] = (tau[:, LEG] ** 2).sum(1); df["tau2_arm"] = (tau[:, ARM] ** 2).sum(1); df["tau2_waist"] = (tau[:, WAIST] ** 2).sum(1)
df["mech"] = (tau * dq).sum(1); df["mech_pos"] = np.clip(tau * dq, 0, None).sum(1)
df["absdq_arm"] = np.abs(dq[:, ARM]).sum(1); df["absdq_leg"] = np.abs(dq[:, LEG]).sum(1)
picks = [("2026-11-18", "Arm\\Heart - Run 1"), ("2026-11-18", "Arm\\Reject - Run 1"), ("2026-11-18", "Arm\\Stand Still - Run 1"),
         ("2026-11-13", "stand still - Run 1 - exec157031804"), ("2026-11-13", "hug - Run 1 - exec157032846"),
         ("2026-11-18", "Locomotion\\Walk forward (Run 1)"), ("2026-11-18", "Locomotion\\Step in place (Run 1)"),
         ("2026-11-06", "Walk forward - Run 1"), ("2026-08-04", "squat"), ("2026-08-04", "heart")]
fig, axes = plt.subplots(len(picks), 3, figsize=(20, 3.0 * len(picks)))
for r, (d, s) in enumerate(picks):
    g = df[(df.date == d) & (df.seg == s)]
    t = g.t - g.t.iloc[0]
    ax = axes[r, 0]
    ax.plot(t, g.P_bms, label="P_bms (V*I)", lw=1.5); ax.plot(t, g.P_mb, label="P_mb (v0*i)", lw=1, alpha=.8)
    for f in g.fsm.unique():
        if f != 802: ax.fill_between(t, 0, 1, where=(g.fsm == f), transform=ax.get_xaxis_transform(), alpha=.15, label=f"fsm {f}")
    hold = g.arm_holding.fillna(False).astype(bool)
    if hold.any(): ax.fill_between(t, 0, 1, where=hold, transform=ax.get_xaxis_transform(), alpha=.08, color="g", label="arm holding")
    ax.set_title(f"{d} | {s}", fontsize=9); ax.legend(fontsize=7); ax.set_ylabel("W")
    ax = axes[r, 1]
    ax.plot(t, g.tau2_leg, label="sum tau^2 legs"); ax.plot(t, g.tau2_arm, label="sum tau^2 arms"); ax.plot(t, g.tau2_waist, label="waist")
    ax.legend(fontsize=7); ax.set_ylabel("N2m2")
    ax = axes[r, 2]
    ax.plot(t, g.mech, label="sum tau*dq"); ax.plot(t, g.absdq_arm * 5, label="5*sum|dq| arms"); ax.plot(t, g.absdq_leg * 5, label="5*sum|dq| legs")
    ax2 = ax.twinx(); ax2.plot(t, g.maxT, "k:", label="max motor T"); ax2.set_ylabel("degC")
    ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "diag_timeseries.png"), dpi=70)
print("saved")
