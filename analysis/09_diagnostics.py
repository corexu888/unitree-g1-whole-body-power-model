"""Model B diagnostics: coefficients + stability, hold-out splits, damping check, residual structure."""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1power as g
ROOT = sys.argv[1]; A = os.path.join(ROOT, "analysis")
df = g.load(ROOT)
pd.set_option("display.width", 250, "display.max_rows", 400, "display.max_columns", 60)
SPEC = dict(cu="type", mech="split"); KW = dict(fixed={"mech_pos": 1.0}, upper={"mech_neg": 1.0})
X = g.features(df, SPEC); X.insert(0, "const", 1.0)
meta_all, Xf_all = g.bms_align(df, X)
damp = (meta_all.fsm == 1).values
meta, Xf = meta_all[~damp], Xf_all[~damp]; y = meta.P_bms.values
coef = g.fit_bounded(Xf, y, **KW)
pred_cv, C = g.lobo_cv(meta, Xf, y, meta.boot.values, **KW)

print("== 1. coefficients: all-data fit, leave-one-boot-out range, segment-cluster bootstrap 95% ==")
rng = np.random.default_rng(0)
segs = meta.segid.values; useg = pd.unique(segs); idx_by = {s: np.where(segs == s)[0] for s in useg}
boots = []
for b in range(300):
    pick = rng.choice(useg, len(useg), replace=True)
    ii = np.concatenate([idx_by[s] for s in pick])
    boots.append(g.fit_bounded(Xf.iloc[ii], y[ii], **KW))
Bs = pd.DataFrame(boots)
tab = pd.DataFrame({"coef": coef, "lobo_min": C.min(1), "lobo_max": C.max(1), "bs_2.5%": Bs.quantile(.025), "bs_97.5%": Bs.quantile(.975)})
# typical contribution: coef * mean regressor (W) for standing / arm hold / walking rows
loco = meta.action.isin(g.LOCO).values
tab["W_mean_arm_rows"] = coef * Xf[~loco].mean(); tab["W_mean_loco_rows"] = coef * Xf[loco].mean(); tab["W_p99_all"] = coef * Xf.quantile(.99)
print(tab.round(4).to_string())
tab.to_csv(os.path.join(A, "coef_table.csv"), encoding="utf-8-sig")

print("\n== 2. damping (fsm=1, tau~0) rows, never used in fitting ==")
md = meta_all[damp]; pd_ = Xf_all[damp].values @ coef.values
t = pd.DataFrame({"date": md.date.values, "seg": md.seg.values, "P_bms": md.P_bms.values, "pred": pd_, "P_mb": md.P_mb.values,
                  "tau2sum": Xf_all[damp][[c for c in Xf_all.columns if c.startswith("cu_")]].sum(1).values, "t": md.t.values})
t["trel"] = t.groupby("seg").t.transform(lambda x: x - x.min())
print(t.round(2).to_string())

print("\n== 3. hold-out splits (train -> test), errors on test rows ==")
def split(name, tr, te):
    c = g.fit_bounded(Xf[tr], y[tr], **KW); p = Xf[te].values @ c.values
    s = g.summarize(meta[te], p, y[te], name); s["n_tr"] = int(tr.sum()); s["n_te"] = int(te.sum()); s["P0"] = c["const"]
    return s, c
isA = (meta.robot == "A").values; arm = ~loco
rows = []
rows.append(split("B -> A (other robot)", ~isA, isA)[0])
rows.append(split("B -> A excluding squat", ~isA, isA & (meta.action != "squat").values)[0])
rows.append(split("arm+stand only -> locomotion", arm, loco)[0])
rows.append(split("locomotion only -> arm+stand", loco, arm)[0])
rows.append(split("dates <= 11-13 -> 11-18", (meta.date <= "2026-11-13").values, (meta.date == "2026-11-18").values)[0])
rows.append(split("11-18 only -> all earlier", (meta.date == "2026-11-18").values, (meta.date < "2026-11-18").values)[0])
for d in sorted(meta.date.unique()):
    rows.append(split(f"leave date {d} out", (meta.date != d).values, (meta.date == d).values)[0])
for a in sorted(meta.action.unique()):
    rows.append(split(f"leave action '{a}' out", (meta.action != a).values, (meta.action == a).values)[0])
S = pd.DataFrame(rows)
print(S[["model", "n_tr", "n_te", "P0", "rmse", "mae", "bias", "seg_mae", "seg_max", "r2"]].round(2).to_string(index=False))
S.to_csv(os.path.join(A, "holdout_splits.csv"), index=False, encoding="utf-8-sig")

print("\n== 4. LOBO-CV residual (pred - meas) structure ==")
e = pred_cv - y
R = meta[["date", "boot", "robot", "action", "segid", "soc", "soh", "volt_v", "maxT", "tick", "mb_temperature0", "P_bms", "P_mb", "t"]].copy()
R["e"] = e; R["pred"] = pred_cv
Tm = meta[[f"T{j}" for j in range(29)]].values
R["T_ankle"] = Tm[:, [4, 5, 10, 11]].max(1); R["T_arm"] = Tm[:, g.ARM].max(1); R["uptime_min"] = R.tick / 60000
print("by boot:"); print(R.groupby(["boot", "date", "robot"]).agg(bias=("e", "mean"), rmse=("e", lambda x: np.sqrt((x ** 2).mean())), n=("e", "size"),
      V=("volt_v", "mean"), soh=("soh", "first"), T_ankle=("T_ankle", "mean"), off_bms_mb=("P_bms", "mean")).assign(
      off_bms_mb=R.groupby(["boot", "date", "robot"]).apply(lambda x: (x.P_bms - x.P_mb).mean())).round(2).to_string())
for c in ["volt_v", "soc", "T_ankle", "T_arm", "maxT", "uptime_min", "mb_temperature0", "P_bms"]:
    q = pd.qcut(R[c], 5, duplicates="drop")
    print(f"\nresidual by {c} quintile (static+arm rows only):")
    print(R[~loco].groupby(q[~loco], observed=True).e.agg(["mean", "std", "size"]).round(2).T.to_string())
seg = R.groupby(["segid", "date", "boot", "action"]).agg(meas=("P_bms", "mean"), pred=("pred", "mean"), e=("e", "mean"), n=("e", "size"),
      Tank=("T_ankle", "max"), V=("volt_v", "mean")).reset_index()
seg["pct"] = seg.e / seg.meas * 100
print("\nsegment-mean error distribution (LOBO-CV, 111 segments): abs W  p50 %.2f p90 %.2f p95 %.2f max %.2f | abs %%  p50 %.2f p90 %.2f max %.2f"
      % (seg.e.abs().median(), seg.e.abs().quantile(.9), seg.e.abs().quantile(.95), seg.e.abs().max(), seg.pct.abs().median(), seg.pct.abs().quantile(.9), seg.pct.abs().max()))
print("worst 12 segments:"); print(seg.reindex(seg.e.abs().sort_values(ascending=False).index).head(12).round(2).to_string(index=False))
seg.to_csv(os.path.join(A, "segment_errors_cv.csv"), index=False, encoding="utf-8-sig")
R.to_pickle(os.path.join(A, "resid_cv.pkl"))
json.dump({"spec": SPEC, "coef": coef.to_dict(), "bms_fir": list(map(float, g.BMS_FIR))}, open(os.path.join(A, "model_B_raw.json"), "w"), indent=1)
