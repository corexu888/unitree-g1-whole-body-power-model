"""Orin step-load test in the damping state (robot B, 23 Nov 2026): which onboard power sensor sees the computer's load?
Usage: python 28_orin_load_test.py <project root> <telemetry.jsonl of the test>
Phases stated by the tester (45 s each): idle in low-power mode, CPU stress, CPU + GPU stress, back to idle;
Orin input power read from tegrastats by the tester: about 8.5, 17.5, 28.5 and 8.8 W (not in the telemetry file)."""
import sys, os, json
import numpy as np, pandas as pd
ROOT, PATH = sys.argv[1], sys.argv[2]; A = os.path.dirname(os.path.abspath(__file__))          # outputs stay in this folder, not in the main line
pd.set_option("display.width", 220, "display.max_rows", 200)
M = json.load(open(os.path.join(ROOT, "g1_power_model_v2.json"), encoding="utf-8"))
rows = []
with open(PATH, encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line: continue
        o = json.loads(line); b = o["bms"]; mb = o.get("mainboard") or {}; mo = o["lowstate"]["motors"]; v = mb.get("value_raw") or [np.nan] * 3
        rows.append(dict(t=o["time_unix"], fsm=(o.get("fsm") or {}).get("id"), I_bms=abs(b["current_a"]), cur_raw=b["current_raw"], V_bms=b["voltage_v"], soc=b["soc_percent"], soh=b["soh_percent"],
                         MV0=v[0], MV1=v[1], I_mb=v[2], mbT=(mb.get("temperature_raw") or [np.nan])[0], fan=sum(mb.get("fan_state_raw") or [0]), mb_age=o.get("mainboard_age_seconds"),
                         bms_age=o.get("bms_age_seconds"), bmsT=np.mean([x for x in b["temperature_raw"] if x > 0]), tau_max=max(abs(m["tau_est"]) for m in mo), speed=sum(abs(m["dq"]) for m in mo),
                         Tmot=np.mean([m["temperature"] for m in mo]), Tmot_max=max(m["temperature"] for m in mo), Vmot=np.mean([m["vol"] for m in mo if m["vol"] > 1]),
                         imuT=(o.get("secondary_imu") or {}).get("temperature")))
d = pd.DataFrame(rows); d["ts"] = d.t - d.t.iloc[0]; d["P_bms"] = d.V_bms * d.I_bms; d["P_mb"] = d.MV0 * d.I_mb; d["dP"] = d.P_bms - d.P_mb
print("frames %d, %.1f s, %.2f Hz; fsm ids %s; SOH %s, SOC %.0f -> %.0f %%; pack voltage %.2f -> %.2f V" % (len(d), d.ts.iloc[-1], len(d) / d.ts.iloc[-1], sorted(d.fsm.dropna().unique().tolist()),
      sorted(d.soh.unique().tolist()), d.soc.iloc[0], d.soc.iloc[-1], d.V_bms.iloc[:20].mean(), d.V_bms.iloc[-20:].mean()))
print("joints: largest |tau| %.2f N*m, largest sum|dq| %.2f rad/s; main-board message age max %.3f s; BMS value changes every %.2f s" % (d.tau_max.max(), d.speed.max(), d.mb_age.max(), d.ts.iloc[-1] / (d.cur_raw.diff() != 0).sum()))
print("model v2 for these frames (zero torque): P0 = %.1f W" % M["P0"])

print("\n== 5 s bins ==")
d["bin"] = (d.ts // 5).astype(int) * 5
b5 = d.groupby("bin").agg(P_bms=("P_bms", "mean"), I_bms=("I_bms", "mean"), P_mb=("P_mb", "mean"), I_mb=("I_mb", "mean"), dP=("dP", "mean"), V_bms=("V_bms", "mean"), MV0=("MV0", "mean"), mbT=("mbT", "mean"), Tmot=("Tmot", "mean"), fan=("fan", "max"))
print(b5.round(2).to_string())

# step times from the BMS power itself: largest changes between consecutive 2 s means
d["b2"] = (d.ts // 2).astype(int) * 2; m2 = d.groupby("b2").P_bms.mean(); ch = m2.diff().abs().sort_values(ascending=False).head(6)
print("\nlargest changes of the 2 s mean BMS power, at (s):", {int(k): round(float(m2.diff()[k]), 1) for k in ch.index})

print("\n== phase means (first 10 s and last 3 s of each stated phase left out) ==")
PH = [("P1 idle, low-power mode", 0, 45, 8.5), ("P2 CPU stress", 45, 90, 17.5), ("P3 CPU + GPU stress", 90, 135, 28.5), ("P4 back to idle", 135, 180, 8.8)]
out = []
for name, a, e, vdd in PH:
    x = d[(d.ts >= a + 10) & (d.ts < e - 3)]
    out.append(dict(phase=name, orin_VDD_IN_stated=vdd, n=len(x), P_bms=x.P_bms.mean(), P_bms_sd=x.P_bms.std(), P_mb=x.P_mb.mean(), P_mb_sd=x.P_mb.std(), I_bms=x.I_bms.mean(), I_mb=x.I_mb.mean(),
                    dP=x.dP.mean(), V_bms=x.V_bms.mean(), bms_minus_MV0=(x.V_bms - x.MV0).mean(), bms_minus_Vmot=(x.V_bms - x.Vmot).mean(), mbT=x.mbT.mean(), Tmot=x.Tmot.mean()))
O = pd.DataFrame(out).set_index("phase"); print(O.round(3).to_string())
O.round(4).to_csv(os.path.join(A, "orin_load_test_phases.csv"), encoding="utf-8-sig")
ref = O.iloc[[0, 3]].mean()          # idle before and after
print("\n== steps relative to the mean of the two idle phases ==")
for name in O.index[1:3]:
    r = O.loc[name]; dv = r.orin_VDD_IN_stated - ref.orin_VDD_IN_stated; db = r.P_bms - ref.P_bms; dm = r.P_mb - ref.P_mb
    print("%-22s Orin input +%.1f W (stated) | BMS +%.2f W (ratio %.2f) | main board %+.2f W (ratio %.2f) | BMS - main board %+.2f W" % (name, dv, db, db / dv, dm, dm / dv, r.dP - ref.dP))
print("idle after minus idle before: BMS %+.2f W, main board %+.2f W, offset %+.2f W" % (O.P_bms.iloc[3] - O.P_bms.iloc[0], O.P_mb.iloc[3] - O.P_mb.iloc[0], O.dP.iloc[3] - O.dP.iloc[0]))
# regression over the four phases: BMS and main-board power against the stated Orin input power
xs = O.orin_VDD_IN_stated.values
for nm in ["P_bms", "P_mb", "dP"]:
    c = np.polyfit(xs, O[nm].values, 1); print("   %-6s = %.2f + %.3f * VDD_IN  (4 phase means)" % (nm, c[1], c[0]))
print("\noffset at Orin idle %.1f W; at CPU stress %.1f W; at CPU + GPU stress %.1f W. Offsets in the nine test sessions: 8.9-24.5 W (robot B, without 6 Nov: 18.1-24.5 W)" % (ref.dP, O.dP.iloc[1], O.dP.iloc[2]))
print("zero-torque power on the BMS: idle %.1f W, CPU stress %.1f W, CPU + GPU stress %.1f W; model constant %.1f W" % (ref.P_bms, O.P_bms.iloc[1], O.P_bms.iloc[2], M["P0"]))

# ---- figure: the two sensors during the four phases (2 s means)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
BLUE, ORANGE = "#2a78d6", "#eb6834"; INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
                     "axes.edgecolor": AXIS, "axes.linewidth": 0.7, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 300, "pdf.fonttype": 42})
OUT = os.path.join(os.path.dirname(A), "figures"); os.makedirs(OUT, exist_ok=True)
m = d.groupby("b2").agg(t=("ts", "mean"), P_bms=("P_bms", "mean"), P_mb=("P_mb", "mean"))
fig, ax = plt.subplots(figsize=(5.2, 2.7))
for a_ in (45, 90, 135): ax.axvline(a_, color=AXIS, lw=0.7)
ax.plot(m.t, m.P_bms, color=BLUE, lw=1.5, label="BMS"); ax.plot(m.t, m.P_mb, color=ORANGE, lw=1.5, label="main board")
for (name, a_, e_, vdd), lab in zip(PH, ["computer idle", "CPU load", "CPU and GPU load", "computer idle"]):
    ax.text((a_ + e_) / 2, 129.5, f"{lab}\n{vdd:.1f} W", ha="center", va="top", fontsize=7.5, color=INK2)
ax.set_xlim(0, 180); ax.set_ylim(80, 131); ax.set_xlabel("time (s)"); ax.set_ylabel("power (W)"); ax.grid(axis="x", visible=False)
ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.33))
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig7_orin_load_test.png"), bbox_inches="tight"); fig.savefig(os.path.join(OUT, "fig7_orin_load_test.pdf"), bbox_inches="tight"); plt.close(fig)
print("saved fig7_orin_load_test")
