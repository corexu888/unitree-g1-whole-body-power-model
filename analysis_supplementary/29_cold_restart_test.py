"""Cold-restart test (robot B, 23 Nov 2026): same battery, robot powered off and on several times, 30 s in the damping state each time.
Question: does the main-board reading shift between power-ons, as it appeared to have done on 6 Nov (voltage +0.3 V, offset about 10 W lower)?
Usage: python 29_cold_restart_test.py <project root> <telemetry.jsonl of the test>"""
import sys, os, json
import numpy as np, pandas as pd
ROOT, PATH = sys.argv[1], sys.argv[2]; A = os.path.dirname(os.path.abspath(__file__))          # outputs stay in this folder, not in the main line
pd.set_option("display.width", 250, "display.max_rows", 200, "display.max_columns", 40)
rows = []
with open(PATH, encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line: continue
        o = json.loads(line); b = o.get("bms") or {}; mb = o.get("mainboard") or {}; ls = o.get("lowstate") or {}; mo = ls.get("motors") or []
        v = mb.get("value_raw") or [np.nan] * 3
        rows.append(dict(t=o["time_unix"], fsm=(o.get("fsm") or {}).get("id"), tick=ls.get("tick"), I_bms=abs(b.get("current_a", np.nan)), cur_raw=b.get("current_raw"), V_bms=b.get("voltage_v", np.nan),
                         soc=b.get("soc_percent"), soh=b.get("soh_percent"), MV0=v[0], MV1=v[1], I_mb=v[2], mbT=(mb.get("temperature_raw") or [np.nan])[0],
                         ls_age=o.get("lowstate_age_seconds"), bms_age=o.get("bms_age_seconds"), mb_age=o.get("mainboard_age_seconds"),
                         tau_max=max([abs(m["tau_est"]) for m in mo], default=np.nan), speed=sum(abs(m["dq"]) for m in mo) if mo else np.nan,
                         Tmot=np.mean([m["temperature"] for m in mo]) if mo else np.nan, Vmot=np.mean([m["vol"] for m in mo if m["vol"] > 1]) if mo else np.nan))
d = pd.DataFrame(rows); d["ts"] = d.t - d.t.iloc[0]; d["P_bms"] = d.V_bms * d.I_bms; d["P_mb"] = d.MV0 * d.I_mb; d["dP"] = d.P_bms - d.P_mb; d["dI"] = d.I_bms - d.I_mb
print("frames %d over %.1f s; fsm ids %s; SOH %s; SOC %s" % (len(d), d.ts.iloc[-1], d.fsm.value_counts(dropna=False).to_dict(), sorted(d.soh.dropna().unique().tolist()), sorted(d.soc.dropna().unique().tolist())))
dt = d.t.diff(); print("frame interval: median %.3f s; gaps longer than 2 s: %s" % (dt.median(), [(round(float(d.ts[i]), 1), round(float(dt[i]), 1)) for i in dt.index[dt > 2]]))
print("message ages (s), max: lowstate %.2f, BMS %.2f, main board %.2f" % (d.ls_age.max(), d.bms_age.max(), d.mb_age.max()))
# power-on sessions: the uptime counter restarts, or the log has a gap
new = (d.tick.diff() < 0) | (dt > 2); d["boot"] = new.cumsum()
print("uptime counter at the first frame of each block (s):", (d.groupby("boot").tick.first() / 1000).round(1).to_dict())
fresh = (d.ls_age < 0.5) & (d.bms_age < 1.0) & (d.mb_age < 0.5)
print("frames with fresh data from all three sources: %d of %d" % (fresh.sum(), len(d)))

print("\n== per power-on: all frames (first 5 s left out) ==")
out = []
for b, x in d[fresh].groupby("boot"):
    x = x[x.t - x.t.iloc[0] >= 5.0]
    if len(x) < 30:
        print("   block", b, "skipped: only", len(x), "frames after the first 5 s"); continue
    out.append(dict(power_on=b + 1, start_s=x.ts.iloc[0], dur_s=x.ts.iloc[-1] - x.ts.iloc[0], frames=len(x), uptime_s=x.tick.iloc[0] / 1000, fsm=x.fsm.mode().iloc[0], soc=x.soc.mean(), V_bms=x.V_bms.mean(), MV0=x.MV0.mean(), MV1=x.MV1.mean(), Vmot=x.Vmot.mean(),
                    bms_minus_MV0=(x.V_bms - x.MV0).mean(), bms_minus_MV1=(x.V_bms - x.MV1).mean(), bms_minus_Vmot=(x.V_bms - x.Vmot).mean(), I_bms=x.I_bms.mean(), I_mb=x.I_mb.mean(), dI=x.dI.mean(),
                    P_bms=x.P_bms.mean(), P_bms_sd=x.P_bms.std(), P_mb=x.P_mb.mean(), P_mb_sd=x.P_mb.std(), dP=x.dP.mean(), mbT=x.mbT.mean(), Tmot=x.Tmot.mean(), tau_max=x.tau_max.max(), speed_max=x.speed.max()))
O = pd.DataFrame(out).set_index("power_on")
print(O[["start_s", "dur_s", "frames", "uptime_s", "fsm", "soc", "V_bms", "MV0", "MV1", "Vmot", "mbT", "Tmot", "tau_max"]].round(3).to_string())
print(O[["bms_minus_MV0", "bms_minus_MV1", "bms_minus_Vmot", "I_bms", "I_mb", "dI", "P_bms", "P_bms_sd", "P_mb", "P_mb_sd", "dP"]].round(3).to_string())
O.round(4).to_csv(os.path.join(A, "cold_restart_test.csv"), encoding="utf-8-sig")
print("\nspread over the power-ons (max - min): BMS - MV0 %.3f V; BMS - MV1 %.3f V; main-board current %.3f A; BMS power %.2f W; main-board power %.2f W; offset %.2f W" % (
    np.ptp(O.bms_minus_MV0), np.ptp(O.bms_minus_MV1), np.ptp(O.I_mb), np.ptp(O.P_bms), np.ptp(O.P_mb), np.ptp(O.dP)))
ref = pd.read_csv(os.path.join(ROOT, "analysis", "offset_forensics_sessions.csv"), index_col=0)
print("for comparison, the nine test sessions: BMS - MV0 was %.3f V on 6 Nov and %.3f to %.3f V in the other robot-B sessions; offset %.1f W on 6 Nov, %.1f to %.1f W otherwise" % (
    ref.loc[2, "bms_minus_MV0"], ref.loc[3:, "bms_minus_MV0"].min(), ref.loc[3:, "bms_minus_MV0"].max(), ref.loc[2, "dP"], ref.loc[3:, "dP"].min(), ref.loc[3:, "dP"].max()))
print("one main-board ADC step is 0.0387 V; one current step is 0.05 A (about %.1f W at %.0f V)" % (0.05 * O.MV0.mean(), O.MV0.mean()))
# drift inside each recording: first 10 s against last 10 s
print("\n== inside each recording: last 10 s minus first 10 s after the 5 s lead-in ==")
for b, x in d[fresh].groupby("boot"):
    x = x[x.t - x.t.iloc[0] >= 5.0]
    if len(x) < 150: continue
    a_, e_ = x[x.t - x.t.iloc[0] < 10], x[x.t.iloc[-1] - x.t < 10]
    print("   power-on %d: BMS %+.2f W, main board %+.2f W, offset %+.2f W, main-board temperature %+.1f K" % (b + 1, e_.P_bms.mean() - a_.P_bms.mean(), e_.P_mb.mean() - a_.P_mb.mean(), e_.dP.mean() - a_.dP.mean(), e_.mbT.mean() - a_.mbT.mean()))
