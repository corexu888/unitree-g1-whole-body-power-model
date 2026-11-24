"""Compare candidate model structures with leave-one-boot-out cross-validation.
Target: BMS pack power (V*|I|) at fresh BMS samples; regressors passed through the BMS moving-average response."""
import sys, os
import numpy as np, pandas as pd
from scipy.optimize import nnls
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1power as g
ROOT = sys.argv[1]
df = g.load(ROOT)
pd.set_option("display.width", 250, "display.max_rows", 300, "display.max_columns", 60)

def fit_nnls(X, y, free=("const",)):
    """Non-negative least squares; columns named in `free` are sign-free."""
    cols = list(X.columns); A = X.values.astype(float)
    sc = np.sqrt((A ** 2).mean(0)); sc[sc == 0] = 1
    As = A / sc
    ext = [As]; 
    for c in free:
        if c in cols: ext.append(-As[:, [cols.index(c)]])
    B = np.hstack(ext)
    w, _ = nnls(B, y, maxiter=20000)
    coef = w[:len(cols)].copy(); k = len(cols)
    for c in free:
        if c in cols: coef[cols.index(c)] -= w[k]; k += 1
    return pd.Series(coef / sc, index=cols)

SPECS = {
 "M0 const": None,
 "M1 single tau2": dict(cu="all"),
 "M2 tau2 x3 groups": dict(cu="group"),
 "M3 tau2 x16 types": dict(cu="type"),
 "M3j tau2 x29 joints": dict(cu="joint"),
 "M4 types+mech net": dict(cu="type", mech="net"),
 "M4s types+mech split": dict(cu="type", mech="split"),
 "M5 types+mech split+|dq|": dict(cu="type", mech="split", fric="abs"),
 "M5b types+mech split+|dq|+dq2": dict(cu="type", mech="split", fric="abs+sq"),
 "M6 types+|dq| (no mech)": dict(cu="type", fric="abs"),
 "M7 groups+mech split+|dq|": dict(cu="group", mech="split", fric="abs"),
}
res = []; store = {}
for name, spec in SPECS.items():
    X = g.features(df, spec) if spec else pd.DataFrame(index=df.index)
    X.insert(0, "const", 1.0)
    meta, Xf = g.bms_align(df, X)
    keep = (meta.fsm != 1).values                      # damping frames kept out of this comparison
    meta, Xf = meta[keep], Xf[keep]
    y = meta.P_bms.values
    coef = fit_nnls(Xf, y)
    pred_in = Xf.values @ coef.values
    # leave-one-boot-out
    pred = np.zeros(len(y))
    for b in sorted(meta.boot.unique()):
        te = (meta.boot == b).values
        c = fit_nnls(Xf[~te], y[~te])
        pred[te] = Xf[te].values @ c.values
    e = pred - y
    segerr = pd.Series(e).groupby(meta.segid.values).mean()
    booterr = pd.Series(e).groupby(meta.boot.values).mean()
    res.append(dict(model=name, k=Xf.shape[1], rmse_in=np.sqrt(np.mean((pred_in - y) ** 2)), rmse_cv=np.sqrt(np.mean(e ** 2)), mae_cv=np.mean(np.abs(e)),
                    seg_mae=segerr.abs().mean(), seg_max=segerr.abs().max(), boot_bias_min=booterr.min(), boot_bias_max=booterr.max(),
                    r2_cv=1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)))
    store[name] = (meta, Xf, coef, pred)
print("rows (fresh BMS samples, damping excluded):", len(y), " boots:", meta.boot.nunique(), " segments:", meta.segid.nunique())
print(pd.DataFrame(res).round(2).to_string(index=False))
for nm in ["M3 tau2 x16 types", "M5 types+mech split+|dq|", "M7 groups+mech split+|dq|"]:
    meta, Xf, coef, pred = store[nm]
    print(f"\n== {nm}: coefficients (all data) ==")
    print(coef.round(4).to_string())
    e = pred - meta.P_bms.values
    t = pd.DataFrame({"e": e, "boot": meta.boot.values, "date": meta.date.values, "action": meta.action.values})
    print("LOBO-CV bias by boot:"); print(t.groupby(["boot", "date"]).e.agg(["mean", "std", "size"]).round(2).to_string())
    print("LOBO-CV bias by date x action:"); print(t.groupby(["date", "action"]).e.agg(["mean", lambda x: np.sqrt(np.mean(x ** 2)), "size"]).round(2).to_string())
