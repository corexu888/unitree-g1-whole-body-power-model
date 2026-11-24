"""Load every telemetry jsonl into one flat frame-level table (analysis/frames.pkl)
and a per-segment summary (analysis/segments.csv). No filtering here."""
import json, glob, os, re, sys
import numpy as np, pandas as pd
ROOT = sys.argv[1]
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "analysis")

def norm_action(seg):
    s = seg.lower()
    s = re.sub(r"\(run \d+\).*", "", s)
    s = re.sub(r"- run \d+.*", "", s)
    s = re.sub(r"- repeat count \d+", "", s)
    s = s.replace("_", " ").strip(" -")
    s = {"handsup": "hands up", "high five": "high five"}.get(s, s)
    return s

def run_no(seg):
    m = re.search(r"run (\d+)", seg.lower())
    return int(m.group(1)) if m else 1

rows = []
files = sorted(glob.glob(os.path.join(DATA, "**", "*telemetry.jsonl"), recursive=True))
JN = None
for f in files:
    rel = os.path.relpath(f, DATA)
    parts = rel.split(os.sep)
    batch = parts[0]
    seg = os.sep.join(parts[1:-1])
    segname = parts[-2]
    date = batch[:10]
    robot = "A" if "robotA" in batch else "B"
    m = re.search(r"exec(\d+)", segname)
    with open(f, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line: continue
            o = json.loads(line)
            ls = o["lowstate"]; b = o["bms"]; mb = o.get("mainboard") or {}
            imu = o.get("secondary_imu") or {}
            aas = o.get("arm_action_state") or {}
            aj = aas.get("json") or {}
            r = dict(batch=batch, date=date, robot=robot, seg=seg, action=norm_action(segname),
                     run=run_no(segname), exec_id=(m.group(1) if m else ""), frame=i,
                     t=o["time_unix"], ls_t=o["lowstate_timestamp"], tick=ls["tick"],
                     ls_age=o["lowstate_age_seconds"], bms_age=o["bms_age_seconds"],
                     mb_age=o.get("mainboard_age_seconds"),
                     soc=b["soc_percent"], soh=b["soh_percent"], cur_raw=b["current_raw"],
                     cur_a=b["current_a"], volt_v=b["voltage_v"], pack_w=b["pack_power_w"],
                     vraw0=b["voltage_raw"][0], vraw1=b["voltage_raw"][1], vraw2=b["voltage_raw"][2],
                     mode_machine=ls["mode_machine"], mode_pr=ls["mode_pr"],
                     maxT=ls["max_motor_temperature"], maxT_joint=ls["max_motor_temperature_joint"],
                     mech_sum=ls["mechanical_power_sum_w"], mech_abs=ls["mechanical_power_abs_sum_w"],
                     fsm=o["fsm"]["id"], fsm_ok=o["fsm"]["ok"], monitor=o["monitor_state"],
                     profile=o.get("profile"), arm_id=aj.get("id"), arm_name=aj.get("name"),
                     arm_holding=aj.get("holding"))
            for k, v in enumerate(b["temperature_raw"]): r[f"bmsT{k}"] = v
            for key in ("value_raw", "temperature_raw", "fan_state_raw", "state_raw"):
                arr = mb.get(key) or [np.nan] * 6
                for k, v in enumerate(arr): r[f"mb_{key[:-4]}{k}"] = v
            for key in ("rpy", "gyroscope", "accelerometer"):
                arr = imu.get(key) or [np.nan] * 3
                for k, v in enumerate(arr): r[f"imu_{key[:3]}{k}"] = v
            names = [mo["name"] for mo in ls["motors"]]
            if JN is None: JN = names
            assert names == JN, (f, i)
            for j, mo in enumerate(ls["motors"]):
                r[f"tau{j}"] = mo["tau_est"]; r[f"dq{j}"] = mo["dq"]
                r[f"q{j}"] = mo.get("q", np.nan); r[f"ddq{j}"] = mo.get("ddq", np.nan)
                r[f"T{j}"] = mo["temperature"]; r[f"vol{j}"] = mo["vol"]
                r[f"ms{j}"] = mo["motorstate_raw"]
            rows.append(r)
df = pd.DataFrame(rows)
df.to_pickle(os.path.join(OUT, "frames.pkl"))
with open(os.path.join(OUT, "joint_names.json"), "w") as fh: json.dump(JN, fh, indent=0)
print("joints:", JN)
print(df.shape)
g = df.groupby(["batch", "seg"], sort=False)
s = g.agg(action=("action", "first"), run=("run", "first"), n=("t", "size"),
          t0=("t", "min"), t1=("t", "max"), soc0=("soc", "first"), soc1=("soc", "last"),
          soh=("soh", "first"), V=("volt_v", "mean"), I=("cur_a", "mean"), P=("pack_w", "mean"),
          Pmin=("pack_w", "min"), Pmax=("pack_w", "max"), maxT=("maxT", "max"),
          fsm=("fsm", lambda x: ",".join(str(v) for v in pd.unique(x))),
          mm=("mode_machine", lambda x: ",".join(str(v) for v in pd.unique(x))),
          prof=("profile", lambda x: ",".join(str(v) for v in pd.unique(x))),
          mon=("monitor", lambda x: ",".join(str(v) for v in pd.unique(x))),
          arm=("arm_name", lambda x: ",".join(str(v) for v in pd.unique(x))))
s["dur"] = s.t1 - s.t0
s["hz"] = (s.n - 1) / s.dur
s["start_utc"] = pd.to_datetime(s.t0, unit="s").dt.strftime("%m-%d %H:%M:%S")
s.to_csv(os.path.join(OUT, "segments.csv"), encoding="utf-8-sig")
pd.set_option("display.width", 250, "display.max_rows", 200, "display.max_colwidth", 60)
print(s[["action", "run", "n", "start_utc", "dur", "hz", "soc0", "soc1", "soh", "V", "I", "P", "maxT", "fsm", "mm"]].round(2).to_string())
