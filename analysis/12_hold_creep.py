"""Hold-phase creep: does measured power drift upward during a static hold while the model stays flat? Relation to motor heating."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1power as g
ROOT = sys.argv[1]; A = os.path.join(ROOT, "analysis")
df = g.load(ROOT)
pd.set_option("display.width", 250, "display.max_rows", 400)
SPEC = dict(cu="type", mech="split"); KW = dict(fixed={"mech_pos": 1.0}, upper={"mech_neg": 1.0})
X = g.features(df, SPEC); X.insert(0, "const", 1.0)
meta, Xf = g.bms_align(df, X); k = (meta.fsm != 1).values; meta, Xf = meta[k], Xf[k]; y = meta.P_bms.values
pred, C = g.lobo_cv(meta, Xf, y, meta.boot.values, **KW)
tau, dq, T = g.arrays(meta)
R = meta[["date", "boot", "action", "segid", "t", "P_bms", "arm_holding"]].copy(); R["e"] = y - pred     # measured - predicted
R["quiet"] = np.abs(dq).sum(1) < 0.6
R["Tarm"] = T[:, g.ARM].max(1); R["Tleg"] = T[:, g.LEG].max(1); R["Twaist"] = T[:, g.WAIST].max(1)
arm_cu = Xf[[f"cu_{t}" for t in g.TYPES[9:]]].values @ C.mean(1)[[f"cu_{t}" for t in g.TYPES[9:]]].values
waist_cu = Xf[[f"cu_{t}" for t in g.TYPES[6:9]]].values @ C.mean(1)[[f"cu_{t}" for t in g.TYPES[6:9]]].values
R["arm_cu"] = arm_cu; R["waist_cu"] = waist_cu
rows = []
for sid, gg in R[(R.arm_holding == True) & R.quiet].groupby("segid"):
    if len(gg) < 12: continue
    gg = gg.sort_values("t"); tt = gg.t.values - gg.t.values[0]
    gg = gg[(tt > 1.0)]; tt = gg.t.values - gg.t.values[0]
    if len(gg) < 10: continue
    sl = np.polyfit(tt, gg.e.values, 1)[0]
    rows.append(dict(date=gg.date.iloc[0], action=gg.action.iloc[0], dur=tt[-1], slope_W_per_s=sl, drift_W=sl * tt[-1], dTarm=gg.Tarm.iloc[-1] - gg.Tarm.iloc[0],
                     dTwaist=gg.Twaist.iloc[-1] - gg.Twaist.iloc[0], arm_cu=gg.arm_cu.mean(), waist_cu=gg.waist_cu.mean(), P=gg.P_bms.mean(), e_mean=gg.e.mean()))
H = pd.DataFrame(rows)
print(H.groupby("action").agg(n=("dur", "size"), dur=("dur", "mean"), drift_W=("drift_W", "mean"), drift_min=("drift_W", "min"), drift_max=("drift_W", "max"),
      dTarm=("dTarm", "mean"), dTwaist=("dTwaist", "mean"), arm_cu=("arm_cu", "mean"), waist_cu=("waist_cu", "mean"), P=("P", "mean")).round(2).to_string())
H["rel"] = H.drift_W / (H.arm_cu + H.waist_cu)
print("\ncorr(drift_W, dTarm) = %.2f ; drift as fraction of (arm+waist copper term) per K of arm heating: median %.4f /K (copper: 0.0039/K)"
      % (np.corrcoef(H.drift_W, H.dTarm)[0, 1], (H.rel / H.dTarm.replace(0, np.nan)).median()))
print("by date for reject:"); print(H[H.action == "reject"].round(2).to_string(index=False))
