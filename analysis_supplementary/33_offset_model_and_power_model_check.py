"""Combines the four supplementary tests (Orin load, cold restarts, battery packs, pack voltage) and asks what they imply:
 A what each test gives for the two sensors
 B an offset model built from the tests alone, used to predict the BMS - main-board offset of the eight robot-B test sessions
 C checks of the power model: the constant term against the zero-torque power, its dependence on computer load, pack and voltage,
   and the held-out session errors after a pack correction
 D figure with the four tests
Usage: python 33_offset_model_and_power_model_check.py <project root> <folder that holds Test1..Test4>
Reads the tables written by scripts 28-30 (this folder) and, read-only, results of the main line (analysis/*.csv, g1_power_model_v2.json)."""
import sys, os, json
import numpy as np, pandas as pd
from scipy import stats
ROOT, BASE = sys.argv[1], sys.argv[2]; A = os.path.dirname(os.path.abspath(__file__)); MAIN = os.path.join(ROOT, "analysis")
pd.set_option("display.width", 250, "display.max_columns", 40)
M = json.load(open(os.path.join(ROOT, "g1_power_model_v2.json"), encoding="utf-8")); P0 = M["P0"]
T1 = pd.read_csv(os.path.join(A, "orin_load_test_phases.csv")); T2 = pd.read_csv(os.path.join(A, "cold_restart_test.csv")); T3 = pd.read_csv(os.path.join(A, "battery_swap_test.csv")); T4 = pd.read_csv(os.path.join(A, "voltage_scan_test.csv"))
def fit(x, y):
    r = stats.linregress(x, y); h = stats.t.ppf(0.975, len(x) - 2) * r.stderr
    return r, "%+.3f (95%% %+.3f to %+.3f, n = %d)" % (r.slope, r.slope - h, r.slope + h, len(x))

print("== A1. computer load (test 1; four phase means; Orin input power as stated by the tester, not in the file) ==")
x = T1.orin_VDD_IN_stated.values
rb, sb = fit(x, T1.P_bms.values); rm, sm = fit(x, T1.P_mb.values); rd, sd_ = fit(x, T1.dP.values)
print("BMS power per W of computer input: %s; intercept %.1f W" % (sb, rb.intercept)); print("main-board power per W: %s" % sm); print("offset per W: %s; intercept %.1f W" % (sd_, rd.intercept))
ORIN_REF = 11.5; print("BMS power expected at the default idle load of %.1f W: %.1f W" % (ORIN_REF, rb.intercept + rb.slope * ORIN_REF))

print("\n== A2. cold restarts (test 2; five power-ons, same pack) ==")
T2["state"] = np.where(T2.bms_minus_MV0 < 0.06, "shifted", "normal"); n_ = T2[T2.state == "normal"]; s_ = T2[T2.state == "shifted"]
print("normal state (%d power-ons): BMS - MV0 %.3f V, main-board current %.3f A, main-board power %.1f W (sd %.2f), offset %.1f W (sd %.2f)" % (len(n_), n_.bms_minus_MV0.mean(), n_.I_mb.mean(), n_.P_mb.mean(), n_.P_mb.std(), n_.dP.mean(), n_.dP.std()))
print("shifted state (%d power-on, no. %s): BMS - MV0 %.3f V, main-board current %.3f A, main-board power %.1f W, offset %.1f W" % (len(s_), s_.power_on.tolist(), s_.bms_minus_MV0.mean(), s_.I_mb.mean(), s_.P_mb.mean(), s_.dP.mean()))
D_STATE = s_.dP.mean() - n_.dP.mean()
print("shifted minus normal: main-board voltage %+.3f V, current %+.3f A, power %+.1f W, offset %+.1f W; BMS power %+.2f W" % (n_.bms_minus_MV0.mean() - s_.bms_minus_MV0.mean(), s_.I_mb.mean() - n_.I_mb.mean(), s_.P_mb.mean() - n_.P_mb.mean(), D_STATE, s_.P_bms.mean() - n_.P_bms.mean()))
print("BMS power over the five power-ons: %.1f-%.1f W, mean %.2f, sd %.2f" % (T2.P_bms.min(), T2.P_bms.max(), T2.P_bms.mean(), T2.P_bms.std()))

print("\n== A3. pack voltage (test 4; one pack, five states of charge) ==")
V = T4.V_bms.values; r_pb, s_pb = fit(V, T4.P_bms.values); r_ib, s_ib = fit(V, T4.I_bms.values); r_im, s_im = fit(V, T4.I_mb.values); r_pm, s_pm = fit(V, T4.P_mb.values); r_dp, s_dp = fit(V, T4.dP.values)
print("pack voltage %.2f-%.2f V" % (V.min(), V.max()))
print("BMS power per V: %s W/V | BMS current per V: %s A/V (constant power of %.0f W: %+.4f A/V)" % (s_pb, s_ib, T4.P_bms.mean(), -T4.P_bms.mean() / V.mean() ** 2))
print("main-board current per V: %s A/V (constant power of %.0f W: %+.4f A/V) | main-board power per V: %s W/V | offset per V: %s W/V" % (s_im, T4.P_mb.mean(), -T4.P_mb.mean() / V.mean() ** 2, s_pm, s_dp))
gap_ref = lambda v: r_dp.intercept + r_dp.slope * np.asarray(v, float)                  # offset of the SOH-98 pack in the normal state at the default idle load

print("\n== A4. battery packs (test 3; three packs, same robot) ==")
T3["dP_at_same_V"] = T3.dP - gap_ref(T3.V_bms) ; T3["pack"] = T3.soh.astype(int)
print(T3[["pack", "V_bms", "P_bms", "P_mb", "dP", "dP_at_same_V"]].round(2).to_string(index=False))
ref98 = T3[T3.pack == 98].iloc[0]; D_PACK_GAP = {int(r.pack): float(r.dP_at_same_V - ref98.dP_at_same_V) for r in T3.itertuples()}; D_PACK_BMS = {int(r.pack): float(r.P_bms - ref98.P_bms) for r in T3.itertuples()}
print("relative to the SOH-98 pack: BMS power %s W; offset at equal voltage %s W; main-board power %s W" % ({k: round(v, 2) for k, v in D_PACK_BMS.items()}, {k: round(v, 2) for k, v in D_PACK_GAP.items()}, {int(r.pack): round(r.P_mb - ref98.P_mb, 2) for r in T3.itertuples()}))
print("offset of the SOH-98 pack in test 3 against the line of test 4 at the same voltage: %+.2f W" % ref98.dP_at_same_V)

print("\n== B. offset model from the tests alone, against the robot-B sessions (offsets there are means over all frames outside the damping state) ==")
S = pd.read_csv(os.path.join(MAIN, "offset_forensics_sessions.csv")); S = S[S.robot == "B"].copy(); S["pack"] = S.soh.astype(int)
S["state"] = np.where(S.bms_minus_MV0 < 0.06, "shifted", "normal"); S["pack_tested"] = S.pack.isin(D_PACK_GAP)
S["pred"] = gap_ref(S.V_bms) + S.pack.map(D_PACK_GAP).fillna(0.0) + np.where(S.state == "shifted", D_STATE, 0.0); S["err"] = S.pred - S.dP
S["pred_voltage_only"] = gap_ref(S.V_bms); S["pred_no_state"] = gap_ref(S.V_bms) + S.pack.map(D_PACK_GAP).fillna(0.0)
print(S[["session", "date", "pack", "pack_tested", "state", "V_bms", "dP", "pred", "err"]].round(2).to_string(index=False))
k = S[S.pack_tested]; sst = ((k.dP - k.dP.mean()) ** 2).sum()
print("sessions with a tested pack (n = %d): observed offset %.1f-%.1f W (sd %.1f); mean absolute error of the prediction %.2f W, largest %.2f W, bias %+.2f W; share of the between-session variance explained %.2f" % (
    len(k), k.dP.min(), k.dP.max(), k.dP.std(), k.err.abs().mean(), k.err.abs().max(), k.err.mean(), 1 - (k.err ** 2).sum() / sst))
for lab, c in [("pack voltage only", "pred_voltage_only"), ("pack voltage + pack", "pred_no_state")]:
    e = k[c] - k.dP; print("   %-22s mean absolute error %.2f W, variance explained %.2f" % (lab, e.abs().mean(), 1 - (e ** 2).sum() / sst))
kn = k[k.state == "normal"]; sstn = ((kn.dP - kn.dP.mean()) ** 2).sum()
print("   normal-state sessions with a tested pack only (n = %d): observed sd %.1f W, mean absolute error %.2f W, largest %.2f W, variance explained %.2f" % (len(kn), kn.dP.std(), kn.err.abs().mean(), kn.err.abs().max(), 1 - (kn.err ** 2).sum() / sstn))
print("terms of the offset model: offset of the SOH-98 pack in the normal state at 52.0 V %.2f W; per volt %.2f W/V; pack %s W; shifted state %+.2f W" % (float(gap_ref(52.0)), r_dp.slope, {k_: round(v, 2) for k_, v in D_PACK_GAP.items()}, D_STATE))
u = S[~S.pack_tested]
for r in u.itertuples(): print("session %d used a pack that was not in test 3 (SOH %d): observed %.1f W, predicted without a pack term %.1f W, difference %+.1f W" % (r.session, r.pack, r.dP, r.pred, r.dP - r.pred))
S.round(3).to_csv(os.path.join(A, "session_offset_prediction.csv"), index=False, encoding="utf-8-sig")

print("\n== C1. constant term of the power model against the zero-torque BMS power ==")
z = pd.concat([T2.assign(test="2 restarts", pack=98)[["test", "pack", "V_bms", "P_bms"]], T3.assign(test="3 packs")[["test", "pack", "V_bms", "P_bms"]], T4.assign(test="4 voltage", pack=98)[["test", "pack", "V_bms", "P_bms"]]])
for p_, x_ in z.groupby("pack"): print("pack SOH %d: %d recordings, BMS power %.1f-%.1f W, mean %.2f W (sd %.2f); minus the model constant %.2f W: %+.2f W" % (p_, len(x_), x_.P_bms.min(), x_.P_bms.max(), x_.P_bms.mean(), x_.P_bms.std() if len(x_) > 1 else np.nan, P0, x_.P_bms.mean() - P0))
print("model constant %.2f W (95%% interval 105.1-106.6 W from resampling segments, 103.8-107.5 W from resampling sessions; main line)" % P0)
print("computer: BMS power at 8.5 / 17.5 / 28.5 / 8.8 W of stated input: %s W -> the constant term holds for the computer load of the test sessions and changes by %.2f W per W" % (T1.P_bms.round(1).tolist(), rb.slope))

print("\n== C2. held-out session errors of the main line after a pack correction (predicted minus measured, W) ==")
E = pd.read_csv(os.path.join(MAIN, "rev_nested_interval.csv")); ps = pd.read_csv(os.path.join(MAIN, "rev_per_session.csv")); sess = pd.read_csv(os.path.join(MAIN, "offset_forensics_sessions.csv"))
E["session"] = E.boot + 1; E["pack"] = E.session.map(sess.set_index("session").soh).astype(int)
chk = E.groupby("boot").err.mean().round(3).values; print("check against the main-line table: per-session bias identical: %s" % bool(np.allclose(chk, ps.bias.values, atol=2e-3)))
def summary(e, by):
    g_ = pd.DataFrame(dict(e=e, s=by)).groupby("s").e
    return e.abs().mean(), g_.apply(lambda x_: x_.abs().mean()).mean(), e.abs().max(), g_.mean()
variants = {"none (main line)": {}, "SOH-95 pack only": {95: D_PACK_BMS[95]}, "SOH-95 and SOH-100 packs": {95: D_PACK_BMS[95], 100: D_PACK_BMS[100]}}
rows = []
for lab, corr in variants.items():
    e = E.err + E.pack.map(corr).fillna(0.0); mae, mae_s, mx, bias = summary(e, E.session)
    rows.append(dict(correction=lab, mae_segments=mae, mae_sessions=mae_s, largest=mx, bias_s5=bias[5], bias_s8=bias[8], bias_s9=bias[9], bias_min=bias.min(), bias_max=bias.max()))
C = pd.DataFrame(rows); print(C.round(2).to_string(index=False)); C.round(3).to_csv(os.path.join(A, "pack_correction_sessions.csv"), index=False, encoding="utf-8-sig")
print("pack per session:", E.groupby("session").pack.first().to_dict(), "| the correction adds the difference of the pack's zero-torque BMS power from that of the SOH-98 pack (test 3): %s W" % {k_: round(v, 2) for k_, v in D_PACK_BMS.items()})

# ---------------------------------------------------------------- D. figure
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
BLUE, ORANGE = "#2a78d6", "#eb6834"; INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
                     "axes.edgecolor": AXIS, "axes.linewidth": 0.7, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 300, "pdf.fonttype": 42})
rows = []
with open(os.path.join(BASE, "Test1", "unitree_robot_telemetry.jsonl"), encoding="utf-8") as fh:
    for line in fh:
        if not line.strip(): continue
        o = json.loads(line); v = (o.get("mainboard") or {}).get("value_raw") or [np.nan] * 3
        rows.append((o["time_unix"], o["bms"]["voltage_v"] * abs(o["bms"]["current_a"]), v[0] * v[2]))
d = pd.DataFrame(rows, columns=["t", "P_bms", "P_mb"]); d["ts"] = d.t - d.t.iloc[0]; m = d.groupby((d.ts // 2).astype(int)).agg(t=("ts", "mean"), P_bms=("P_bms", "mean"), P_mb=("P_mb", "mean"))
fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.0)); YL = (78, 134)
a = ax[0, 0]
for x_ in (45, 90, 135): a.axvline(x_, color=AXIS, lw=0.7)
a.plot(m.t, m.P_bms, color=BLUE, lw=1.5); a.plot(m.t, m.P_mb, color=ORANGE, lw=1.5)
for (s0, s1), lab in zip([(0, 45), (45, 90), (90, 135), (135, 180)], ["8.5 W", "17.5 W", "28.5 W", "8.8 W"]): a.text((s0 + s1) / 2, 132.5, lab, ha="center", va="top", fontsize=7.5, color=INK2)
a.text(4, 106.5, "BMS", color=BLUE, fontsize=8, fontweight="bold"); a.text(4, 90.5, "main board", color=ORANGE, fontsize=8, fontweight="bold")
a.set_xlim(0, 180); a.set_xlabel("time (s); computer input power above"); a.set_title("(a) computer load", loc="left", fontsize=8.5)
a = ax[0, 1]
a.plot(T2.power_on, T2.P_bms, "o", color=BLUE, ms=5); a.plot(T2.power_on, T2.P_mb, "o", color=ORANGE, ms=5)
sx = int(s_.power_on.iloc[0]); a.annotate("main-board voltage\nreads 0.30 V higher", (sx, s_.P_mb.iloc[0]), xytext=(sx + 0.35, 99.0), fontsize=7.5, color=INK2, ha="left", va="center", arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.7, shrinkB=5))
a.set_xticks(T2.power_on); a.set_xlim(0.5, 5.5); a.set_xlabel("power-on number (same pack)"); a.set_title("(b) cold restarts", loc="left", fontsize=8.5)
a = ax[1, 0]
xs = np.arange(len(T3)); a.plot(xs, T3.P_bms, "o", color=BLUE, ms=5); a.plot(xs, T3.P_mb, "o", color=ORANGE, ms=5)
for x_, r in zip(xs, T3.itertuples()): a.text(x_, r.P_bms + 2.2, "%.1f" % r.P_bms, ha="center", fontsize=7.5, color=INK2); a.text(x_, r.P_mb - 4.6, "%.1f" % r.P_mb, ha="center", fontsize=7.5, color=INK2)
a.set_xticks(xs); a.set_xticklabels(["%d%%" % p_ for p_ in T3.pack]); a.set_xlim(-0.6, len(T3) - 0.4); a.set_xlabel("battery pack (state of health reported by its BMS)"); a.set_title("(c) battery packs", loc="left", fontsize=8.5)
a = ax[1, 1]
vv = np.linspace(V.min() - 0.1, V.max() + 0.1, 20)
a.plot(vv, r_pb.intercept + r_pb.slope * vv, color=BLUE, lw=1.0, alpha=0.6); a.plot(vv, r_pm.intercept + r_pm.slope * vv, color=ORANGE, lw=1.0, alpha=0.6)
a.plot(V, T4.P_bms, "o", color=BLUE, ms=5); a.plot(V, T4.P_mb, "o", color=ORANGE, ms=5)
a.set_xlabel("pack voltage (V)"); a.set_title("(d) pack voltage", loc="left", fontsize=8.5)
for a in ax.ravel(): a.set_ylim(*YL); a.grid(axis="x", visible=False)
for a in ax[:, 0]: a.set_ylabel("power in the zero-torque state (W)")
fig.tight_layout(); OUT = os.path.join(os.path.dirname(A), "figures")
fig.savefig(os.path.join(OUT, "fig7_supplementary_tests.png"), bbox_inches="tight"); fig.savefig(os.path.join(OUT, "fig7_supplementary_tests.pdf"), bbox_inches="tight"); plt.close(fig)
print("\nsaved figures/fig7_supplementary_tests")
