"""Checks of an engineer's analysis of the BMS - main-board offset (9-25 W), on the raw telemetry.
A voltages per session: BMS pack voltage, the two main-board voltage readings, the bus voltage reported by the motors
B the damping phase at the end of the aborted hug execution: does the main-board reading freeze, and is it stale?
C is the offset continuous across a restart with the same battery?        D drift of the offset with pack voltage inside a session
E offset and model bias per session against the battery health reading    F how fresh the main-board data are in general"""
import sys, os, json
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1power as g
ROOT = sys.argv[1]; A = os.path.join(ROOT, "analysis")
pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_rows", 300)
df = g.load(ROOT); tau, dq, T = g.arrays(df)
segi = df.groupby("segid", sort=False).cumcount().values; K = len(g.BMS_FIR) - 1
vol = df[[c for c in df.columns if c.startswith("vol") and c[3:].isdigit()]].values.astype(float)
df["Vmot"] = np.nanmean(np.where(vol > 1, vol, np.nan), axis=1)
df["I_bms"] = df.cur_a.abs(); df["I_mb"] = df.mb_value2
for c in ["P_mb", "I_mb"]:
    df[c + "_f"] = df.groupby("segid")[c].transform(lambda s: np.convolve(s.values, g.BMS_FIR, mode="full")[:len(s)]); df.loc[segi < K, c + "_f"] = np.nan
fresh = df.groupby("segid").cur_raw.transform(lambda s: np.r_[False, np.diff(s.values) != 0]).values.astype(bool) & df.P_mb_f.notna().values
nd = (df.fsm != 1).values
d = df[fresh & nd].copy(); d["dP"] = d.P_bms - d.P_mb_f; d["dI"] = d.I_bms - d.I_mb_f

print("== A. voltages per session (means over all non-damping frames; V) ==")
v = df[nd].groupby("boot").agg(date=("date", "first"), robot=("robot", "first"), soh=("soh", "median"), soc_first=("soc", "first"), soc_last=("soc", "last"), V_bms=("volt_v", "mean"), MV0=("mb_value0", "mean"), MV1=("mb_value1", "mean"), V_motor=("Vmot", "mean"))
v["bms_minus_MV0"] = v.V_bms - v.MV0; v["bms_minus_MV1"] = v.V_bms - v.MV1; v["bms_minus_motor"] = v.V_bms - v.V_motor; v["MV0_over_bms"] = v.MV0 / v.V_bms
v["dP"] = d.groupby("boot").dP.mean(); v["dI"] = d.groupby("boot").dI.mean()
v.index = v.index + 1; v.index.name = "session"
print(v.round(4).to_string())
steps = np.sort(df.mb_value0.diff().abs().replace(0, np.nan).dropna().unique())[:3]
print("smallest steps of the main-board voltage reading:", np.round(steps, 5), "| 3.3/4096*48 =", round(3.3 / 4096 * 48, 5))
v.round(5).to_csv(os.path.join(A, "offset_forensics_sessions.csv"), encoding="utf-8-sig")

print("\n== B. end of the aborted hug execution (exec157032846): last frames ==")
x = df[df.segid.str.contains("exec157032846")].sort_values("t").copy()
x["ts"] = x.t - x.t.iloc[0]; x["tau_max"] = np.abs(tau[x.index]).max(1); x["speed"] = np.abs(dq[x.index]).sum(1); x["Tmax"] = T[x.index].max(1)
dm = x[x.fsm == 1]
print("frames %d, damping frames %d, damping from %.2f s to %.2f s" % (len(x), len(dm), dm.ts.min(), dm.ts.max()))
cols = ["ts", "fsm", "P_bms", "I_bms", "volt_v", "P_mb", "I_mb", "mb_value0", "mb_age", "bms_age", "tau_max", "speed", "Tmax"]
print(x[cols].iloc[-(len(dm) + 12):].iloc[::3].round(3).to_string(index=False))
print("in damping: main-board current values seen:", sorted(dm.I_mb.round(3).unique().tolist()), "| main-board voltage values:", len(dm.mb_value0.unique()), "distinct | mb_age min %.3f max %.3f s | BMS power first %.1f last %.1f W" % (dm.mb_age.min(), dm.mb_age.max(), dm.P_bms.iloc[0], dm.P_bms.iloc[-1]))
print("joint speed in damping: mean sum|dq| %.2f rad/s, max %.2f; pack voltage first %.2f last %.2f V" % (dm.speed.mean(), dm.speed.max(), dm.volt_v.iloc[0], dm.volt_v.iloc[-1]))
# all damping frames in the data set
allm = df[df.fsm == 1]
print("all damping frames in the data: %d, in segments: %s" % (len(allm), allm.groupby("segid").size().to_dict()))

print("\n== C. offset across the restart between sessions 6 and 7 (same day; one row per segment, in time order) ==")
sg = d.groupby("segid").agg(boot=("boot", "first"), t0=("t", "min"), soh=("soh", "median"), soc=("soc", "mean"), V=("volt_v", "mean"), dP=("dP", "mean"), dI=("dI", "mean"), P_bms=("P_bms", "mean"), uptime_min=("tick", lambda s: s.mean() / 60000)).sort_values("t0")
sg["session"] = sg.boot + 1
y = sg[sg.session.isin([6, 7])]; cut = (y.session == 7).values.argmax()
print(y.iloc[max(0, cut - 4):cut + 5][["session", "soh", "soc", "V", "uptime_min", "dP", "dI"]].round(3).to_string())

print("\n== D. offset against pack voltage inside each session (one point per segment) ==")
rows = []
for s_, x_ in sg.groupby("session"):
    if len(x_) < 5: continue
    r = stats.linregress(x_.V, x_.dI); r2 = stats.linregress(x_.V, x_.dP)
    rows.append(dict(session=s_, segments=len(x_), V_min=x_.V.min(), V_max=x_.V.max(), dI_per_V=r.slope, r=r.rvalue, dP_per_V=r2.slope))
D = pd.DataFrame(rows); print(D.round(4).to_string(index=False))
w = sg.copy(); w[["Vc", "dIc", "dPc"]] = w[["V", "dI", "dP"]] - w.groupby("session")[["V", "dI", "dP"]].transform("mean")
w = w[w.groupby("session").V.transform("size") >= 5]; r = stats.linregress(w.Vc, w.dIc)
print("pooled within-session slope: %.4f A/V (95%% %.4f to %.4f), r = %.2f, n = %d; a constant-power load of 105 W at 52 V changes its current by %.4f A/V, one of 19 W by %.4f A/V" % (
    r.slope, r.slope - 1.96 * r.stderr, r.slope + 1.96 * r.stderr, r.rvalue, len(w), -105 / 52 ** 2, -19 / 52 ** 2))
# does the BMS current itself follow the constant-power expectation? use quiet arms-down standing in the shallow stance, where the load is nearly constant
print("slope of dP on V, pooled within session: %.2f W/V" % stats.linregress(w.Vc, w.dPc).slope)

print("\n== E. per session: battery health reading, offset, and held-out bias of the power model ==")
ps = pd.read_csv(os.path.join(A, "rev_per_session.csv"), index_col=0)
e = v[["date", "robot", "soh", "V_bms", "dP"]].copy(); e["model_bias"] = ps.bias.values; e["segments"] = ps.segments.values
print(e.round(2).to_string())
print("soh values per session (min-max):", df.groupby("boot").soh.agg(["min", "max"]).apply(lambda r_: f"{r_['min']:.0f}-{r_['max']:.0f}", axis=1).tolist())

print("\n== G. robot A in the damping state (squat segment), in four time blocks ==")
xa = df[df.segid == "2026-08-04|squat"].sort_values("t").copy(); xa["ts"] = xa.t - xa.t.iloc[0]
da = xa[xa.fsm == 1]; da = da.assign(block=pd.cut(da.ts, 4, labels=False))
print(da.groupby("block").agg(t0=("ts", "min"), t1=("ts", "max"), P_bms=("P_bms", "mean"), P_mb=("P_mb", "mean"), V=("volt_v", "mean")).round(2).to_string())

print("\n== H. how much of the offset (current difference per segment) battery and pack voltage explain, robot B sessions 3-9 ==")
bq = sg[sg.session >= 3]
def r2_of(Xm, yv):
    Xm = np.column_stack([Xm, np.ones(len(yv))]); c_, *_ = np.linalg.lstsq(Xm, yv, rcond=None); res = yv - Xm @ c_
    return 1 - (res ** 2).sum() / ((yv - yv.mean()) ** 2).sum(), res.std()
d_soh = pd.get_dummies(bq.soh.astype(int), drop_first=True).values.astype(float); d_ses = pd.get_dummies(bq.session, drop_first=True).values.astype(float)
for label, Xm in [("pack voltage only", bq[["V"]].values), ("battery (health reading) only", d_soh), ("battery + pack voltage", np.column_stack([d_soh, bq.V.values])),
                  ("session only", d_ses), ("session + pack voltage", np.column_stack([d_ses, bq.V.values]))]:
    a_, s_ = r2_of(Xm, bq.dI.values); print("   dI ~ %-30s R2 = %.3f, residual sd %.3f A (%.2f W at 52 V), n = %d segments" % (label, a_, s_, s_ * 52, len(bq)))
print(bq.groupby(["soh", "session"]).agg(n=("dI", "size"), V=("V", "mean"), dI=("dI", "mean"), dP=("dP", "mean")).round(3).to_string())

print("\n== F. freshness of the two sensors (age of the latest message when a frame was logged, s) ==")
print(df[["mb_age", "bms_age"]].describe(percentiles=[.5, .99]).round(3).T.to_string())
print("share of frames with a main-board message older than 0.3 s: %.2f%%" % (100 * (df.mb_age > 0.3).mean()))
