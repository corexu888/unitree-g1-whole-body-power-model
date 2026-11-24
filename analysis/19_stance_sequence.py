import sys, os
sys.path.insert(0, os.path.join(sys.argv[1], "analysis"))
import numpy as np, pandas as pd
import g1power as g
pd.set_option("display.width", 250, "display.max_rows", 400)
df = g.load(sys.argv[1])
dq = df[[f"dq{j}" for j in range(29)]].values
df["quiet"] = (np.abs(dq).sum(1) < 0.5) & (df.fsm != 1)
df["qa"] = df[["q4", "q10"]].mean(1); df["qk"] = df[["q3", "q9"]].mean(1); df["ta"] = df[["tau4", "tau10"]].mean(1)
df["i"] = df.groupby("segid").cumcount()
first = df[(df.i >= 8) & (df.i < 40) & df.quiet]          # first ~3 s of each segment, standing
s = first.groupby("segid").agg(date=("date", "first"), boot=("boot", "first"), t0=("t", "min"), action=("action", "first"), run=("run", "first"),
                               qa=("qa", "mean"), qk=("qk", "mean"), ta=("ta", "mean"), P=("P_bms", "mean"), n=("t", "size")).sort_values("t0")
s["stance"] = np.where(s.qa.isna(), "?", np.where(s.qa < -0.2, "D", "s"))
s["gap_min"] = s.groupby("boot").t0.diff() / 60
print("stance at the start of each segment, in time order (D = deep, ankle < -0.20 rad; s = shallow):")
for b, x in s.groupby("boot"):
    print(f"boot {b} {x.date.iloc[0]}: " + "".join(x.stance) + f"   runs: {x.run.astype(str).str.cat()}  n={len(x)}")
print("\n2026-11-13 by run (round) and boot:")
x = s[s.date == "2026-11-13"]
print(x.groupby(["boot", "run"]).agg(n=("qa", "size"), deep=("stance", lambda v: (v == "D").sum()), qa=("qa", "mean"), ta=("ta", "mean"), P=("P", "mean")).round(3).to_string())
print("\nall dates by run:")
print(s.groupby(["date", "boot", "run"]).agg(n=("qa", "size"), deep=("stance", lambda v: (v == "D").sum()), qa=("qa", "mean"), P=("P", "mean")).round(3).to_string())
# does the stance change inside a segment? compare the first 3 s with the last 3 s of quiet standing
last = df[df.quiet].groupby("segid").tail(30).groupby("segid").agg(qa_end=("qa", "mean"))
c = s.join(last); c["end"] = np.where(c.qa_end < -0.2, "D", "s")
print("\nstance at start vs end of the same segment:"); print(pd.crosstab(c.stance, c.end).to_string())
print("\nsegments where the stance differs from the previous segment of the same boot:")
s["prev"] = s.groupby("boot").stance.shift(); ch = s[(s.stance != s.prev) & s.prev.notna() & (s.stance != "?")]
print(ch[["date", "boot", "action", "run", "prev", "stance", "gap_min", "P"]].round(2).to_string())
