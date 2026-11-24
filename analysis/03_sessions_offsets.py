"""Boot sessions (from tick = ms since boot), battery identity, BMS-vs-mainboard offset and lag."""
import sys, os
import numpy as np, pandas as pd
ROOT = sys.argv[1]; A = os.path.join(ROOT, "analysis")
df = pd.read_pickle(os.path.join(A, "frames.pkl")).sort_values("t").reset_index(drop=True)
pd.set_option("display.width", 250, "display.max_rows", 300)
df["P_bms"] = df.volt_v * df.cur_a.abs()
df["P_mb"] = df.mb_value0 * df.mb_value2
df["boot_t"] = df.t - df.tick / 1000.0
# a new boot whenever the implied boot time jumps by > 5 s
jump = df.boot_t.diff().abs() > 5
df["boot"] = jump.cumsum()
b = df.groupby("boot").agg(date=("date", "first"), robot=("robot", "first"),
        boot_utc=("boot_t", lambda x: pd.to_datetime(x.median(), unit="s").strftime("%m-%d %H:%M:%S")),
        first=("t", lambda x: pd.to_datetime(x.min(), unit="s").strftime("%H:%M:%S")),
        last=("t", lambda x: pd.to_datetime(x.max(), unit="s").strftime("%H:%M:%S")),
        up0_min=("tick", lambda x: x.min() / 60000), up1_min=("tick", lambda x: x.max() / 60000),
        nseg=("seg", "nunique"), n=("t", "size"), soh=("soh", lambda x: sorted(x.unique())),
        soc0=("soc", "first"), soc1=("soc", "last"), drift=("boot_t", lambda x: x.max() - x.min()),
        off_W=("P_bms", "mean"), )
b["off_W"] = df.groupby("boot").apply(lambda x: (x.P_bms - x.P_mb).mean())
b["off_A"] = df.groupby("boot").apply(lambda x: (x.cur_a.abs() - x.mb_value2).mean())
b["Tmb"] = df.groupby("boot").mb_temperature0.mean()
print(b.round(2).to_string())
df[["boot"]].to_pickle(os.path.join(A, "boot.pkl"))

print("\n== offset P_bms - P_mb by boot x soh x action ==")
o = df.groupby(["boot", "soh", "action"]).apply(lambda x: pd.Series({"off_W": (x.P_bms - x.P_mb).mean(),
        "off_A": (x.cur_a.abs() - x.mb_value2).mean(), "V": x.volt_v.mean(), "P_bms": x.P_bms.mean(), "n": len(x)}))
print(o.round(2).to_string())

print("\n== is the offset a constant power or a constant current?  regress (I_bms - I_mb) and (P_bms-P_mb) on V, I, per robot/date ==")
seg = df.groupby(["date", "boot", "seg"]).agg(V=("volt_v", "mean"), I=("cur_a", lambda x: x.abs().mean()), Imb=("mb_value2", "mean"),
        P=("P_bms", "mean"), Pmb=("P_mb", "mean"), soh=("soh", "first"), Tmb=("mb_temperature0", "mean"), act=("action", "first")).reset_index()
seg["offW"] = seg.P - seg.Pmb; seg["offA"] = seg.I - seg.Imb
for d, g in seg.groupby("date"):
    print(d, "n=%d" % len(g), "offW mean %.2f sd %.2f | offA mean %.3f sd %.3f | corr(offW,P)=%.2f corr(offW,V)=%.2f corr(offW,Tmb)=%.2f"
          % (g.offW.mean(), g.offW.std(), g.offA.mean(), g.offA.std(), np.corrcoef(g.offW, g.P)[0, 1], np.corrcoef(g.offW, g.V)[0, 1], np.corrcoef(g.offW, g.Tmb)[0, 1]))

print("\n== lag of BMS power behind mainboard power (cross-correlation on 10 Hz grid, per segment) ==")
def best_lag(x, y, maxlag=30):
    # y(t) ~ x(t - lag); returns lag (frames) maximising correlation
    x = x - x.mean(); y = y - y.mean(); out = []
    for L in range(-10, maxlag + 1):
        if L >= 0: a, c = x[:len(x) - L], y[L:]
        else: a, c = x[-L:], y[:len(y) + L]
        out.append((np.corrcoef(a, c)[0, 1], L))
    return max(out)
rows = []
for (d, s), g in df.groupby(["date", "seg"]):
    if len(g) < 100 or g.P_mb.std() < 3: continue
    r, L = best_lag(g.P_mb.values, g.P_bms.values)
    r0 = np.corrcoef(g.P_mb, g.P_bms)[0, 1]
    rows.append((d, g.action.iloc[0], L, r, r0, g.P_bms.std(), g.P_mb.std()))
lg = pd.DataFrame(rows, columns=["date", "action", "lag_frames", "r_best", "r_lag0", "sd_bms", "sd_mb"])
print(lg.groupby(["date", "action"]).agg(lag=("lag_frames", "median"), lag_min=("lag_frames", "min"), lag_max=("lag_frames", "max"),
      r_best=("r_best", "median"), r0=("r_lag0", "median"), sd_bms=("sd_bms", "median"), sd_mb=("sd_mb", "median"), n=("lag_frames", "size")).round(2).to_string())
