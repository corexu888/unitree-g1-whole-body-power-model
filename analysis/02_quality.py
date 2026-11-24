"""Frame-level quality checks: update rates, staleness, timing, sensor cross-checks."""
import sys, os, json
import numpy as np, pandas as pd
ROOT = sys.argv[1]; A = os.path.join(ROOT, "analysis")
df = pd.read_pickle(os.path.join(A, "frames.pkl"))
pd.set_option("display.width", 250, "display.max_rows", 300)
df["P_bms"] = df.volt_v * df.cur_a.abs()          # discharge-positive, recomputed
print("== sign conventions by batch (mean) ==")
print(df.groupby("date").agg(cur_a=("cur_a", "mean"), pack_w=("pack_w", "mean"), P_bms=("P_bms", "mean"),
      chk=("pack_w", lambda x: 0)).round(2))
print("max |pack_w| - V*|I| :", (df.pack_w.abs() - df.P_bms).abs().max())
print("cur_a vs cur_raw/1000 max diff:", (df.cur_a - df.cur_raw / 1000).abs().max(),
      " volt_v vs vraw0/1000:", (df.volt_v - df.vraw0 / 1000).abs().max())
print("any charging frames (cur>0):", (df.cur_a > 0).sum(), " zero current:", (df.cur_a == 0).sum())

print("\n== update intervals (s) of each source, per batch ==")
def upd(g, cols):
    ch = (g[cols].diff().abs().sum(axis=1) > 0).values
    t = g.t.values[ch]
    return np.diff(t)
rows = []
for (date, seg), g in df.groupby(["date", "seg"], sort=False):
    g = g.sort_values("t")
    for name, cols in [("bms_I", ["cur_raw"]), ("bms_V", ["vraw0"]), ("bms_any", ["cur_raw", "vraw0", "vraw1"]),
                       ("mb_val", ["mb_value0", "mb_value1", "mb_value2"]), ("tick", ["tick"]), ("tau", ["tau0", "tau3", "tau15"])]:
        d = upd(g, cols)
        if len(d): rows.append((date, name, np.median(d), np.percentile(d, 90), len(d) / (g.t.max() - g.t.min())))
u = pd.DataFrame(rows, columns=["date", "src", "med_dt", "p90_dt", "changes_per_s"])
print(u.groupby(["date", "src"]).median().round(3).unstack("src").to_string())

print("\n== ages (s): median / p99 / max ==")
for c in ["ls_age", "bms_age", "mb_age"]:
    print(c, df.groupby("date")[c].agg(["median", lambda x: x.quantile(.99), "max"]).round(3).to_dict("index"))

print("\n== frame timing ==")
df = df.sort_values(["date", "seg", "t"])
df["dt"] = df.groupby(["date", "seg"]).t.diff()
df["dtick"] = df.groupby(["date", "seg"]).tick.diff()
print(df.groupby("date").dt.describe(percentiles=[.5, .99]).round(4))
print("gaps > 0.3 s:", (df.dt > 0.3).sum(), "  dtick<=0:", (df.dtick <= 0).sum())
print(df[df.dt > 0.3][["date", "seg", "frame", "dt", "dtick"]].to_string())
print("tick rate (ticks per second):", (df.dtick / df.dt).describe(percentiles=[.01, .5, .99]).round(1).to_dict())

print("\n== segment time overlaps ==")
s = df.groupby(["date", "seg"]).t.agg(["min", "max"]).reset_index().sort_values("min")
ov = s[s["min"].values < np.r_[-np.inf, np.maximum.accumulate(s["max"].values)[:-1]]]
print("overlapping segments:", len(ov)); print(ov.to_string())

print("\n== fsm / mode / monitor / profile / arm state counts ==")
for c in ["fsm", "mode_machine", "mode_pr", "monitor", "profile", "arm_name", "arm_holding", "fsm_ok"]:
    print(c, df.groupby("date")[c].value_counts(dropna=False).to_dict())
ms = [f"ms{j}" for j in range(29)]
print("motorstate_raw nonzero frames:", (df[ms] != 0).any(axis=1).sum())
print(df.loc[(df[ms] != 0).any(axis=1), ["date", "seg", "frame"] + ms].head(10).to_string())

print("\n== BMS vs mainboard value_raw ==")
for k in range(6):
    print(f"mb_value{k}", df.groupby("date")[f"mb_value{k}"].agg(["min", "median", "max"]).round(3).to_dict("index"))
for k in range(6):
    for nm in ["temperature", "fan_state", "state"]:
        c = f"mb_{nm}{k}"
        v = df[c].dropna()
        if v.nunique() > 1 or (len(v) and v.iloc[0] != 0): print(c, "unique:", sorted(v.unique())[:15], "n=", v.nunique())
df["P_mb0"] = df.mb_value0 * df.mb_value2
df["P_mb1"] = df.mb_value1 * df.mb_value2
g = df.groupby("date")
print(g.apply(lambda x: pd.Series({
    "V_bms": x.volt_v.mean(), "mbv0": x.mb_value0.mean(), "mbv1": x.mb_value1.mean(), "vraw1/1000": x.vraw1.mean() / 1000,
    "motor_vol": x[[f"vol{j}" for j in range(29)]].mean().mean(),
    "I_bms": x.cur_a.abs().mean(), "mbv2": x.mb_value2.mean(),
    "corr(I,mbv2)": np.corrcoef(x.cur_a.abs(), x.mb_value2.fillna(x.mb_value2.mean()))[0, 1],
    "P_bms": x.P_bms.mean(), "P_mb0": x.P_mb0.mean(), "P_bms-P_mb0": (x.P_bms - x.P_mb0).mean()})).round(3).to_string())
print("\nper-action mean, B only:  P_bms, P_mb(v0*i), diff")
print(df.groupby(["date", "action"]).apply(lambda x: pd.Series({"P_bms": x.P_bms.mean(), "P_mb": x.P_mb0.mean(),
      "diff": (x.P_bms - x.P_mb0).mean(), "I_bms": x.cur_a.abs().mean(), "I_mb": x.mb_value2.mean(), "n": len(x)})).round(2).to_string())
print("\nBMS temps / soc / soh unique per date:")
print(df.groupby("date").agg(soh=("soh", lambda x: sorted(x.unique())), soc=("soc", lambda x: (x.min(), x.max())),
      bmsT0=("bmsT0", lambda x: (x.min(), x.max())), bmsT1=("bmsT1", lambda x: (x.min(), x.max())),
      vraw2=("vraw2", lambda x: sorted(x.unique())[:5])).to_string())
