import sys, os, json
sys.path.insert(0, os.path.join(sys.argv[1], "analysis"))
import numpy as np, pandas as pd
import g1power as g
pd.set_option("display.width", 220)
df = g.load(sys.argv[1])
M = json.load(open(os.path.join(sys.argv[1], "g1_power_model_v2.json"), encoding="utf-8"))
tau, dq, T = g.arrays(df)
bj = np.array([M["b"][n.replace("left_", "").replace("right_", "")] for n in g.JN])
df["legspeed"] = np.abs(dq[:, g.LEG]).sum(1)
df["legcu"] = (bj[g.LEG] * tau[:, g.LEG] ** 2).sum(1); df["othercu"] = (bj[g.WAIST + g.ARM] * tau[:, g.WAIST + g.ARM] ** 2).sum(1)
pm = tau * dq; df["mech"] = np.clip(pm, 0, None).sum(1) + M["eta_regen"] * np.clip(pm, None, 0).sum(1)
out = []
for sid, x in df[df.action.isin(g.LOCO)].groupby("segid"):
    x = x.sort_values("t"); mv = (x.legspeed.rolling(10, center=True, min_periods=1).mean() > 1.5).values
    if mv.sum() < 30: continue
    i0, i1 = np.where(mv)[0][0], np.where(mv)[0][-1]
    core = x.iloc[i0 + 12:i1 - 8]                       # drop ~1.2 s after the start (BMS lag) and ~0.8 s before the end
    out.append(dict(segid=sid, action=x.action.iloc[0], date=x.date.iloc[0], move_s=x.t.iloc[i1] - x.t.iloc[i0], core_s=core.t.iloc[-1] - core.t.iloc[0],
                    P_move=core.P_bms.mean(), P_seg=x.P_bms.mean(), legcu=core.legcu.mean(), othercu=core.othercu.mean(), mech=core.mech.mean()))
O = pd.DataFrame(out); print(O.round(1).to_string(index=False))
s = O.groupby("action").agg(n=("P_move", "size"), move_s=("move_s", "mean"), P_move=("P_move", "mean"), P_move_sd=("P_move", "std"), P_seg=("P_seg", "mean"), legcu=("legcu", "mean"), othercu=("othercu", "mean"), mech=("mech", "mean"))
s["P0_share"] = M["P0"] / s.P_move
print(s.round(2).to_string())
w = s.loc["walk forward"]
for m_kg in (35.0,):
    print("walking: mean power while moving %.1f W; command 0.5 m/s; CoT with %.0f kg: total %.2f, above P0 %.2f" % (w.P_move, m_kg, w.P_move / (m_kg * 9.81 * 0.5), (w.P_move - M["P0"]) / (m_kg * 9.81 * 0.5)))
print("segment-mean power range (all fitted segments):", df[df.fsm != 1].groupby("segid").P_bms.mean().quantile([0, 1]).round(1).tolist())
