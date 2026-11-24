"""Explore regressors: ranges per batch, collinearity, single-feature relations."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1power as g
ROOT = sys.argv[1]
df = g.load(ROOT)
pd.set_option("display.width", 250, "display.max_rows", 300, "display.max_columns", 50)
X = g.features(df, dict(cu="type", mech="split", fric="abs+sq"))
print("== mean of each regressor by batch (per-frame) ==")
print(X.groupby(df.date).mean().T.round(2).to_string())
print("\n== 99th percentile by batch ==")
print(X.groupby(df.date).quantile(0.99).T.round(1).to_string())
tau, dq, T = g.arrays(df)
print("\n== per-joint |tau| p50 / p99 / max , |dq| p99, T min-max (all data) ==")
out = pd.DataFrame({"tau_p50": np.percentile(np.abs(tau), 50, 0), "tau_p99": np.percentile(np.abs(tau), 99, 0), "tau_max": np.abs(tau).max(0),
                    "dq_p99": np.percentile(np.abs(dq), 99, 0), "dq_max": np.abs(dq).max(0), "T_min": T.min(0), "T_max": T.max(0),
                    "tau_step": [np.min(np.diff(np.unique(np.round(tau[:, j], 6)))) for j in range(29)]}, index=g.JN)
print(out.round(3).to_string())
print("\n== standing-still segments: leg torques by segment (mean) ==")
ss = df[(df.action == "stand still")]
cols = [f"tau{j}" for j in g.LEG]
tab = ss.groupby(["date", "boot", "seg"])[cols].mean()
tab.columns = [n.replace("left_", "L_").replace("right_", "R_") for n in g.JN[:12]]
tab["P_bms"] = ss.groupby(["date", "boot", "seg"]).P_bms.mean(); tab["maxT"] = ss.groupby(["date", "boot", "seg"]).maxT.max()
print(tab.round(1).to_string())
