"""Two more checks in the damping state (robot B, 23 Nov 2026).
Test 3: three battery packs in the same robot, 30 s each - does the BMS of one pack read higher?
Test 4: one pack at five states of charge - how do the two sensors change with pack voltage?
Usage: python 30_battery_and_voltage_tests.py <project root> <test3 telemetry.jsonl> <test4 telemetry.jsonl>"""
import sys, os, json
import numpy as np, pandas as pd
from scipy import stats
ROOT = sys.argv[1]; A = os.path.dirname(os.path.abspath(__file__))          # outputs stay in this folder, not in the main line
pd.set_option("display.width", 250, "display.max_rows", 200, "display.max_columns", 40)
M = json.load(open(os.path.join(ROOT, "g1_power_model_v2.json"), encoding="utf-8"))

def load(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            o = json.loads(line); b = o.get("bms") or {}; mb = o.get("mainboard") or {}; ls = o.get("lowstate") or {}; mo = ls.get("motors") or []
            v = mb.get("value_raw") or [np.nan] * 3
            rows.append(dict(t=o["time_unix"], fsm=(o.get("fsm") or {}).get("id"), tick=ls.get("tick"), I_bms=abs(b.get("current_a", np.nan)), V_bms=b.get("voltage_v", np.nan), soc=b.get("soc_percent"),
                             soh=b.get("soh_percent"), MV0=v[0], MV1=v[1], I_mb=v[2], mbT=(mb.get("temperature_raw") or [np.nan])[0], ls_age=o.get("lowstate_age_seconds"),
                             bms_age=o.get("bms_age_seconds"), mb_age=o.get("mainboard_age_seconds"), tau_max=max([abs(m["tau_est"]) for m in mo], default=np.nan),
                             speed=sum(abs(m["dq"]) for m in mo) if mo else np.nan, Tmot=np.mean([m["temperature"] for m in mo]) if mo else np.nan,
                             bmsT=np.mean([x for x in (b.get("temperature_raw") or []) if x > 0] or [np.nan])))
    d = pd.DataFrame(rows); d["ts"] = d.t - d.t.iloc[0]; d["P_bms"] = d.V_bms * d.I_bms; d["P_mb"] = d.MV0 * d.I_mb; d["dP"] = d.P_bms - d.P_mb; d["dI"] = d.I_bms - d.I_mb
    dt = d.t.diff(); d["block"] = ((d.tick.diff() < 0) | (dt > 2)).cumsum(); d["new_boot"] = (d.tick.diff() < 0)
    d["fresh"] = (d.ls_age < 0.5) & (d.bms_age < 1.0) & (d.mb_age < 0.5)
    return d

def blocks(d, lead=5.0):
    out = []
    for b, x in d[d.fresh].groupby("block"):
        x = x[x.t - x.t.iloc[0] >= lead]
        if len(x) < 30: print("   block", b + 1, "skipped:", len(x), "frames after the lead-in"); continue
        out.append(dict(block=b + 1, start_s=x.ts.iloc[0], dur_s=x.ts.iloc[-1] - x.ts.iloc[0], uptime_s=x.tick.iloc[0] / 1000, fsm=x.fsm.mode().iloc[0], soh=x.soh.median(), soc=x.soc.mean(),
                        V_bms=x.V_bms.mean(), bms_minus_MV0=(x.V_bms - x.MV0).mean(), bms_minus_MV1=(x.V_bms - x.MV1).mean(), I_bms=x.I_bms.mean(), I_mb=x.I_mb.mean(), dI=x.dI.mean(),
                        P_bms=x.P_bms.mean(), P_bms_sd=x.P_bms.std(), P_mb=x.P_mb.mean(), dP=x.dP.mean(), mbT=x.mbT.mean(), bmsT=x.bmsT.mean(), Tmot=x.Tmot.mean(), tau_max=x.tau_max.max(),
                        P_bms_first10=x[x.t - x.t.iloc[0] < 10].P_bms.mean(), P_bms_last10=x[x.t.iloc[-1] - x.t < 10].P_bms.mean()))
    O = pd.DataFrame(out).set_index("block")
    O["mb_state"] = np.where(O.bms_minus_MV0 < 0.06, "shifted", "normal")
    return O

for name, path in [("Test 3: three battery packs", sys.argv[2]), ("Test 4: one pack at five states of charge", sys.argv[3])]:
    d = load(path)
    print("\n================ %s ================" % name)
    print("frames %d over %.0f s (%.1f min); fsm ids %s; SOH values %s; SOC %.0f-%.0f %%; unix time of first frame %.0f" % (len(d), d.ts.iloc[-1], d.ts.iloc[-1] / 60, d.fsm.value_counts(dropna=False).to_dict(),
          sorted(d.soh.dropna().unique().tolist()), d.soc.min(), d.soc.max(), d.t.iloc[0]))
    gaps = d.t.diff(); print("blocks: %d; gaps (s): %s; power-ons inside the file (uptime counter restarted): %d" % (d.block.nunique(), [round(float(g_), 0) for g_ in gaps[gaps > 2]], int(d.new_boot.sum())))
    O = blocks(d)
    print(O[["start_s", "dur_s", "uptime_s", "fsm", "soh", "soc", "V_bms", "bms_minus_MV0", "bms_minus_MV1", "mb_state", "mbT", "bmsT", "Tmot", "tau_max"]].round(3).to_string())
    print(O[["soh", "soc", "V_bms", "I_bms", "I_mb", "dI", "P_bms", "P_bms_sd", "P_bms_first10", "P_bms_last10", "P_mb", "dP", "mb_state"]].round(3).to_string())
    if "Test 3" in name:
        O.round(4).to_csv(os.path.join(A, "battery_swap_test.csv"), encoding="utf-8-sig")
        print("BMS zero-torque power by pack: " + "; ".join("SOH %.0f%%: %.1f W" % (r.soh, r.P_bms) for r in O.itertuples()) + " | model constant %.1f W" % M["P0"])
        print("spread between packs: BMS %.2f W, main board %.2f W (main-board state: %s)" % (np.ptp(O.P_bms), np.ptp(O.P_mb), O.mb_state.tolist()))
        # 9 sessions for reference
        print("in the nine test sessions the model's held-out bias was -4.0 W with the SOH 95 pack (session 5), -0.4 W with SOH 92, +0.6 to +5.3 W with SOH 98, +0.2 to +0.7 W with SOH 100")
    else:
        O.round(4).to_csv(os.path.join(A, "voltage_scan_test.csv"), encoding="utf-8-sig")
        n = O[O.mb_state == "normal"]
        for lab, x in [("all blocks", O), ("blocks with the main board in its normal state", n)]:
            if len(x) < 3: continue
            r1 = stats.linregress(x.V_bms, x.dI); r2 = stats.linregress(x.V_bms, x.I_bms); r3 = stats.linregress(x.V_bms, x.I_mb); r4 = stats.linregress(x.V_bms, x.P_bms); r5 = stats.linregress(x.V_bms, x.P_mb); r6 = stats.linregress(x.V_bms, x.dP)
            print("%s (n = %d, %.2f-%.2f V):" % (lab, len(x), x.V_bms.min(), x.V_bms.max()))
            print("   current difference: %+.4f A/V (r = %.2f) | BMS current %+.4f A/V | main-board current %+.4f A/V" % (r1.slope, r1.rvalue, r2.slope, r3.slope))
            print("   BMS power %+.2f W/V (r = %.2f) | main-board power %+.2f W/V (r = %.2f) | offset %+.2f W/V (r = %.2f)" % (r4.slope, r4.rvalue, r5.slope, r5.rvalue, r6.slope, r6.rvalue))
            Pm, Vm = x.P_bms.mean(), x.V_bms.mean()
            print("   a constant-power load of %.0f W at %.1f V would change its current by %+.4f A/V; one of %.0f W (main-board share) by %+.4f A/V" % (Pm, Vm, -Pm / Vm ** 2, x.P_mb.mean(), -x.P_mb.mean() / Vm ** 2))
        print("in the nine test sessions: pooled within-session slope of the current difference -0.025 A/V; offset -1.0 W/V")
