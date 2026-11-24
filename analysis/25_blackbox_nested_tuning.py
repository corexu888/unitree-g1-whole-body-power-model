"""Black-box baselines with hyper-parameters chosen inside each training fold (inner grouped cross-validation by session).
Same inputs, rows and splits as 18_innovation_checks.py (F1). Scaling is fitted inside the training fold (pipeline)."""
import sys, os, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1power as g
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, GridSearchCV
ROOT = sys.argv[1]; A = os.path.join(ROOT, "analysis")
pd.set_option("display.width", 230)
df = g.load(ROOT); tau, dq, T = g.arrays(df)
si = df.groupby("segid", sort=False).cumcount().values
Tst = pd.DataFrame(np.where((si < 15)[:, None], T, np.nan)).groupby(df.segid.values).transform("mean").values
XJ = pd.DataFrame(np.column_stack([tau ** 2, tau * dq, np.abs(dq), T - Tst]), index=df.index)
meta_all, Xall = g.bms_align(df, XJ); k = (meta_all.fsm != 1).values
meta, Xbb = meta_all[k], Xall[k].values; y = meta.P_bms.values; segid = meta.segid.values; boots = meta.boot.values
act_of = meta.groupby("segid").action.first()
MODELS = {
 "ridge": (lambda: Pipeline([("s", StandardScaler()), ("m", Ridge())]), {"m__alpha": list(np.logspace(-2, 4, 7))}),
 "gradient boosting": (lambda: HistGradientBoostingRegressor(max_iter=300, random_state=0), {"max_depth": [2, 4], "learning_rate": [0.03, 0.1]}),
 "random forest": (lambda: RandomForestRegressor(n_estimators=100, max_features=0.5, n_jobs=-1, random_state=0), {"min_samples_leaf": [2, 10, 30]}),
 "MLP 64-64": (lambda: Pipeline([("s", StandardScaler()), ("m", MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=400, early_stopping=True, random_state=0))]), {"m__alpha": [1e-3, 0.1, 10.0]}),
}
chosen = {}
def tuned(name):
    make, grid = MODELS[name]
    def f(tr, te):
        gk = GroupKFold(n_splits=min(3, len(set(boots[tr]))))
        gs = GridSearchCV(make(), grid, cv=list(gk.split(Xbb[tr], y[tr], groups=boots[tr])), scoring="neg_mean_absolute_error", n_jobs=1)
        gs.fit(Xbb[tr], y[tr]); chosen.setdefault(name, []).append(str(gs.best_params_)); return gs.predict(Xbb[te])
    return f
gest = ~meta.action.isin(g.LOCO + ["stand still", "squat"]).values; loco = meta.action.isin(g.LOCO).values
def cv(f, groups):
    p = np.zeros(len(y))
    for b in sorted(pd.unique(groups)):
        te = groups == b; p[te] = f(~te, te)
    return p
seg = lambda e, m=None: pd.Series(e if m is None else e[m]).groupby(segid if m is None else segid[m]).mean()
rows = []
for name in MODELS:
    t0 = time.time(); f = tuned(name)
    e1 = seg(cv(f, boots) - y); pa = cv(f, meta.action.values); e2 = seg(pa - y); by = e2.groupby(act_of.reindex(e2.index).values).mean()
    p3 = np.zeros(len(y)); p3[gest] = f(~gest, gest); e3 = seg(p3 - y, gest)
    p4 = np.zeros(len(y)); p4[loco] = f(~loco, loco); e4 = seg(p4 - y, loco)
    rows.append(dict(model=name, session_mae=e1.abs().mean(), session_max=e1.abs().max(), action_mae=e2.abs().mean(), action_max=e2.abs().max(), worst_action=f"{by.abs().idxmax()} {by[by.abs().idxmax()]:+.1f}",
                     nogest_mae=e3.abs().mean(), nogest_max=e3.abs().max(), noloco_mae=e4.abs().mean(), noloco_max=e4.abs().max(), minutes=(time.time() - t0) / 60))
    print(rows[-1], flush=True); print("   most frequent choice:", pd.Series(chosen[name]).value_counts().head(2).to_dict(), flush=True)
    pd.DataFrame(rows).round(3).to_csv(os.path.join(A, "f1_blackbox_nested_tuned.csv"), index=False, encoding="utf-8-sig")
print(pd.DataFrame(rows).round(2).to_string(index=False))
