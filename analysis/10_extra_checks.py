"""Extra robustness checks for model B."""
import sys, os, json, re
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1power as g
ROOT = sys.argv[1]; A = os.path.join(ROOT, "analysis")
df = g.load(ROOT)
pd.set_option("display.width", 250, "display.max_rows", 400, "display.max_columns", 60)
SPEC = dict(cu="type", mech="split"); KW = dict(fixed={"mech_pos": 1.0}, upper={"mech_neg": 1.0})
X = g.features(df, SPEC); X.insert(0, "const", 1.0)
meta, Xf = g.bms_align(df, X); k = (meta.fsm != 1).values; meta, Xf = meta[k], Xf[k]; y = meta.P_bms.values
coef = g.fit_bounded(Xf, y, **KW)
loco = meta.action.isin(g.LOCO).values

print("== a. time-alignment sensitivity: shift the BMS response by s frames (LOBO-CV) ==")
for s in [-2, -1, 0, 1, 2]:
    fir = np.r_[np.zeros(max(s, 0)), g.BMS_FIR] if s >= 0 else g.BMS_FIR[-s:]
    fir = fir / fir.sum()
    m2, X2 = g.bms_align(df, X, fir=fir); kk = (m2.fsm != 1).values; m2, X2 = m2[kk], X2[kk]; y2 = m2.P_bms.values
    p, _ = g.lobo_cv(m2, X2, y2, m2.boot.values, **KW); e = p - y2; lo = m2.action.isin(g.LOCO).values
    print(f" shift {s:+d} frames: rmse arm {np.sqrt((e[~lo]**2).mean()):.2f}  rmse loco {np.sqrt((e[lo]**2).mean()):.2f}  n={len(e)}")

print("\n== b. alternative estimation: steady frames only (no BMS filter), all coefficients vs model B ==")
tau, dq, T = g.arrays(df)
dfs = df.copy(); dfs["act"] = np.abs(dq).sum(1)
dfs["quiet"] = dfs.groupby("segid").act.transform(lambda x: x.rolling(12, min_periods=12).max()) < 0.6     # quiet for the last ~1.2 s
dfs["dP"] = dfs.groupby("segid").P_bms.transform(lambda x: x.rolling(12, min_periods=12).max() - x.rolling(12, min_periods=12).min())
st = (dfs.quiet & (dfs.fsm != 1) & (dfs.dP < 8)).values
fresh = np.r_[False, np.diff(df.cur_raw.values) != 0]
st = st & fresh
Xs = X[st]; ys = df.P_bms.values[st]
cs = g.fit_bounded(Xs, ys, **KW)
print(f" steady fresh rows: {st.sum()} from {df[st].segid.nunique()} segments")
print(pd.DataFrame({"modelB": coef, "steady_only": cs}).round(4).to_string())
ps = Xs.values @ coef.values
print(" model B on steady rows: bias %.2f  rmse %.2f" % ((ps - ys).mean(), np.sqrt(((ps - ys) ** 2).mean())))

print("\n== c. same model on the mainboard reading (10 Hz, per frame, no filter) with one intercept per boot ==")
m = (df.fsm != 1).values & df.P_mb.notna().values
Xm = X[m].copy(); ym = df.P_mb.values[m]
for b in sorted(df.boot.unique()): Xm[f"boot{b}"] = (df.boot.values[m] == b).astype(float)
Xm = Xm.drop(columns="const")
cm = g.fit_bounded(Xm, ym, lower={c: -np.inf for c in Xm.columns if c.startswith("boot")}, **KW)
print(pd.DataFrame({"modelB(BMS)": coef, "mainboard": cm}).round(4).to_string())
em = Xm.values @ cm.values - ym; lo = df.action.isin(g.LOCO).values[m]
print(" frame-level rmse vs mainboard: arm/static %.2f W, locomotion %.2f W" % (np.sqrt((em[~lo] ** 2).mean()), np.sqrt((em[lo] ** 2).mean())))
print(" single-intercept fit to mainboard target -> per-boot bias:")
Xm1 = X[m]; c1 = g.fit_bounded(Xm1, ym, **KW); e1 = Xm1.values @ c1.values - ym
print(pd.Series(e1).groupby(df.boot.values[m]).mean().round(2).to_dict(), " (P0_mb = %.1f)" % c1["const"])
e0 = Xf.values @ coef.values - y
print(" single-intercept fit to BMS target (in-sample) -> per-boot bias:"); print(pd.Series(e0).groupby(meta.boot.values).mean().round(2).to_dict())

print("\n== d. peaks: instantaneous model vs mainboard reading (+ boot offset) ==")
pin = X[m].values @ coef.values                       # instantaneous total-power estimate
off = pd.Series(df.P_bms.values[m] - ym).groupby(df.boot.values[m]).transform("mean").values
pm_tot = ym + off
d = pd.DataFrame({"pin": pin, "pm": pm_tot, "seg": df.segid.values[m], "action": df.action.values[m], "date": df.date.values[m]})
pk = d.groupby(["date", "action", "seg"]).agg(peak_meas=("pm", "max"), peak_pred=("pin", "max"), p99_meas=("pm", lambda x: x.quantile(.99)), p99_pred=("pin", lambda x: x.quantile(.99))).reset_index()
pk["d_peak"] = pk.peak_pred - pk.peak_meas
print(pk.groupby("action").agg(peak_meas=("peak_meas", "mean"), peak_pred=("peak_pred", "mean"), d_mean=("d_peak", "mean"), d_min=("d_peak", "min"), d_max=("d_peak", "max"), n=("seg", "size")).round(1).to_string())
bp = pd.DataFrame({"meas": y, "pred": Xf.values @ coef.values, "seg": meta.segid.values, "action": meta.action.values})
bpk = bp.groupby(["action", "seg"]).agg(meas=("meas", "max"), pred=("pred", "max")).reset_index(); bpk["d"] = bpk.pred - bpk.meas
print("BMS-reading peak per segment (model passed through the BMS response): mean diff by action"); print(bpk.groupby("action").d.agg(["mean", "min", "max"]).round(1).T.to_string())

print("\n== e. window-level errors (LOBO-CV), non-overlapping windows inside segments ==")
R = pd.read_pickle(os.path.join(A, "resid_cv.pkl"))
R["trel"] = R.groupby("segid").t.transform(lambda x: x - x.min())
for w in [2, 5, 10]:
    R["win"] = (R.trel // w).astype(int)
    ww = R.groupby(["segid", "win"]).agg(e=("e", "mean"), n=("e", "size"), meas=("P_bms", "mean"), act=("action", "first")).reset_index()
    ww = ww[ww.n >= w * 1.5]
    lo2 = ww.act.isin(g.LOCO)
    f = lambda s: "mae %.2f p90 %.2f p99 %.2f max %.2f (n=%d)" % (s.abs().mean(), s.abs().quantile(.9), s.abs().quantile(.99), s.abs().max(), len(s))
    print(f" {w:>2d}-s windows | arm/static: {f(ww.e[~lo2])} | locomotion: {f(ww.e[lo2])}")

print("\n== f. June text dump (robot A, other script, ~1 Hz) ==")
def parse(fn):
    out = []; cur = None
    for line in open(fn, encoding="utf-8", errors="replace"):
        line = line.rstrip("\n")
        if line.startswith("=== g1_state_dump"):
            if cur: out.append(cur)
            cur = {}
        elif cur is not None:
            mm = re.match(r"\s+(motor\.tau_est_nm|motor\.dq_rad_s|current_raw|bmsvoltage_raw|tick|soc_percent_raw): (.*)", line)
            if mm:
                key, val = mm.group(1), mm.group(2)
                cur[key] = json.loads(val) if val.startswith("[") else float(val)
            mm = re.match(r"action_status: (.*) \(age", line)
            if mm: cur["status"] = mm.group(1)
    if cur: out.append(cur)
    return out
J = []
for fn in ["heart_monitor.txt", "high_five_monitor.txt"]:
    for i, r in enumerate(parse(os.path.join(ROOT, "data", "2026-06-08_robotA_arm_gestures_text_dump", fn))):
        if "current_raw" not in r or "motor.tau_est_nm" not in r: continue
        ta = np.array(r["motor.tau_est_nm"]); dqa = np.array(r["motor.dq_rad_s"])
        row = {"const": 1.0}
        for t, idx in g.TYPE_IDX.items(): row[f"cu_{t}"] = (ta[idx] ** 2).sum()
        pmm = ta * dqa; row["mech_pos"] = np.clip(pmm, 0, None).sum(); row["mech_neg"] = np.clip(pmm, None, 0).sum()
        pred = sum(coef[c] * row[c] for c in coef.index)
        J.append(dict(file=fn[:10], i=i, status=r.get("status", "")[:38], P_bms=abs(r["current_raw"]) / 1000 * r["bmsvoltage_raw"][0] / 1000, pred_inst=pred,
                      absdq=np.abs(dqa).sum(), soc=r.get("soc_percent_raw")))
J = pd.DataFrame(J); J["e"] = J.pred_inst - J.P_bms
print(J.round(2).to_string(index=False))
q = J[J.absdq < 0.6]
print(" quiet frames only (sum|dq|<0.6): n=%d bias %.2f W, mae %.2f W" % (len(q), q.e.mean(), q.e.abs().mean()))
