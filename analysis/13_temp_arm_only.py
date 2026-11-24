"""Quick test: temperature correction applied only to arm (and waist) copper terms."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1power as g
ROOT = sys.argv[1]
df = g.load(ROOT); tau, dq, T = g.arrays(df)
KW = dict(fixed={"mech_pos": 1.0}, upper={"mech_neg": 1.0})
def run(label, alpha_by_joint, Tref=45.0):
    t2 = tau ** 2 * (1.0 + alpha_by_joint[None, :] * (T - Tref))
    X = pd.DataFrame({f"cu_{t}": t2[:, idx].sum(1) for t, idx in g.TYPE_IDX.items()}, index=df.index)
    pm = tau * dq; X["mech_pos"] = np.clip(pm, 0, None).sum(1); X["mech_neg"] = np.clip(pm, None, 0).sum(1); X.insert(0, "const", 1.0)
    meta, Xf = g.bms_align(df, X); k = (meta.fsm != 1).values; meta, Xf = meta[k], Xf[k]; y = meta.P_bms.values
    p, _ = g.lobo_cv(meta, Xf, y, meta.boot.values, **KW)
    s = g.summarize(meta, p, y, label); e = p - y; lo = meta.action.isin(g.LOCO).values
    hold = (meta.arm_holding == True).values
    print(f"{label:38s} rmse {s['rmse']:.2f} seg_mae {s['seg_mae']:.2f} seg_max {s['seg_max']:.2f} boot [{s['boot_min']:.2f},{s['boot_max']:.2f}] arm-rmse {np.sqrt((e[~lo]**2).mean()):.2f} hold-rmse {np.sqrt((e[hold]**2).mean()):.2f}")
z = np.zeros(29)
run("no temperature term", z)
for a in [0.002, 0.003, 0.0039]:
    v = z.copy(); v[g.ARM] = a; run(f"arms only alpha={a}", v)
    v = z.copy(); v[g.ARM] = a; v[g.WAIST] = a; run(f"arms+waist alpha={a}", v)
