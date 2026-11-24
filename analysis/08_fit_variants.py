"""Second round: physically constrained variants, temperature correction, parallel-mechanism cross terms."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1power as g
ROOT = sys.argv[1]
df = g.load(ROOT)
pd.set_option("display.width", 250, "display.max_rows", 300, "display.max_columns", 60)
tau, dq, T = g.arrays(df)

def build(spec, extra=None):
    X = g.features(df, spec); X.insert(0, "const", 1.0)
    if extra is not None:
        for k, v in extra.items(): X[k] = v
    meta, Xf = g.bms_align(df, X)
    keep = (meta.fsm != 1).values
    return meta[keep], Xf[keep]

rows = []
def run(label, spec, extra=None, **kw):
    meta, Xf = build(spec, extra)
    y = meta.P_bms.values
    pred, C = g.lobo_cv(meta, Xf, y, meta.boot.values, **kw)
    s = g.summarize(meta, pred, y, label); s["k"] = Xf.shape[1] - len(kw.get("fixed", {}) or {})
    # locomotion-only and arm-only segment errors
    e = pd.Series(pred - y); loco = meta.action.isin(g.LOCO).values
    for nm, m in [("loco", loco), ("arm", ~loco)]:
        sg = e[m].groupby(meta.segid.values[m]).mean()
        s[f"segmae_{nm}"] = sg.abs().mean(); s[f"rmse_{nm}"] = np.sqrt((e[m] ** 2).mean())
    rows.append(s)
    return meta, Xf, pred, C

base = dict(cu="type")
run("A  tau2 types", base)
run("B  + mech_pos=1 fixed, regen in [0,1]", dict(cu="type", mech="split"), fixed={"mech_pos": 1.0}, upper={"mech_neg": 1.0})
run("C  + mech_pos>=1, regen in [0,1]", dict(cu="type", mech="split"), lower={"mech_pos": 1.0}, upper={"mech_neg": 1.0})
run("D  + mech free>=0", dict(cu="type", mech="split"))
run("E  B + |dq| groups", dict(cu="type", mech="split", fric="abs"), fixed={"mech_pos": 1.0}, upper={"mech_neg": 1.0})
run("F  B + |dq| + dq2 groups", dict(cu="type", mech="split", fric="abs+sq"), fixed={"mech_pos": 1.0}, upper={"mech_neg": 1.0})
run("G  D + |dq| groups (all free>=0)", dict(cu="type", mech="split", fric="abs"))
run("H  tau2 types + |dq| groups", dict(cu="type", fric="abs"))
for a in [0.002, 0.00393, 0.006, 0.008]:
    run(f"E  alpha={a}", dict(cu="type", mech="split", fric="abs", alpha=a), fixed={"mech_pos": 1.0}, upper={"mech_neg": 1.0})
    run(f"A  alpha={a}", dict(cu="type", alpha=a))
# parallel-mechanism cross terms (ankle pitch*roll per side, waist roll*pitch) - sign-free
cross = {"x_ankleL": tau[:, 4] * tau[:, 5], "x_ankleR": tau[:, 10] * tau[:, 11], "x_waist": tau[:, 13] * tau[:, 14]}
run("E  + ankle/waist cross terms (free sign)", dict(cu="type", mech="split", fric="abs"), extra=cross, fixed={"mech_pos": 1.0}, upper={"mech_neg": 1.0},
    lower={k: -np.inf for k in cross})
out = pd.DataFrame(rows)
print(out[["model", "k", "rmse", "mae", "bias", "seg_mae", "seg_p90", "seg_max", "boot_min", "boot_max", "r2", "segmae_arm", "rmse_arm", "segmae_loco", "rmse_loco"]].round(2).to_string(index=False))
