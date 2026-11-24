"""Estimate how the BMS reading relates in time to the faster mainboard reading:
P_bms(new sample at t) = c_boot + sum_k h_k * P_mb(t - k*0.1 s).  Uses only frames where the BMS content changed."""
import sys, os
import numpy as np, pandas as pd
from scipy.optimize import nnls
ROOT = sys.argv[1]; A = os.path.join(ROOT, "analysis")
df = pd.read_pickle(os.path.join(A, "frames.pkl")).sort_values("t").reset_index(drop=True)
df["boot"] = pd.read_pickle(os.path.join(A, "boot.pkl")).boot.values
df["P_bms"] = df.volt_v * df.cur_a.abs(); df["P_mb"] = df.mb_value0 * df.mb_value2
K = 25
rows = []
for (d, s), g in df.groupby(["date", "seg"], sort=False):
    g = g.sort_values("t")
    if g.P_mb.isna().any(): g = g.assign(P_mb=g.P_mb.interpolate().bfill())
    # resample mainboard power on a uniform 0.1 s grid ending at each frame time (frames are ~0.1033 s apart; use index lags)
    pm = g.P_mb.values; pb = g.P_bms.values; I = g.cur_raw.values; V = g.vraw0.values
    chg = np.r_[False, (np.diff(I) != 0)]           # current content changed -> fresh BMS sample
    dt_ok = np.r_[True, np.diff(g.t.values) < 0.2]
    for i in np.where(chg)[0]:
        if i < K or not dt_ok[i - K:i + 1].all(): continue
        rows.append((g.boot.iloc[0], d, g.action.iloc[0], pb[i]) + tuple(pm[i - k] for k in range(K + 1)))
R = pd.DataFrame(rows, columns=["boot", "date", "action", "y"] + [f"x{k}" for k in range(K + 1)])
print("fresh-BMS rows:", len(R))
X = R[[f"x{k}" for k in range(K + 1)]].values; y = R.y.values
# remove per-boot means (offset differs per boot)
Xc = X - R.groupby("boot")[[f"x{k}" for k in range(K + 1)]].transform("mean").values
yc = y - R.groupby("boot").y.transform("mean").values
for name, m in [("arm batches", R.date.isin(["2026-08-04", "2026-11-11", "2026-11-13", "2026-11-18"]) & ~R.action.isin(["walk forward", "step in place", "turn around"])),
                ("locomotion", R.action.isin(["walk forward", "step in place", "turn around"])), ("all", np.ones(len(R), bool))]:
    m = np.asarray(m)
    h, rn = nnls(Xc[m], yc[m])
    hu = np.linalg.lstsq(Xc[m], yc[m], rcond=None)[0]
    pred = Xc[m] @ h
    print(f"\n[{name}] n={m.sum()}  sum(h)={h.sum():.3f}  centroid lag={np.sum(h*np.arange(K+1))/h.sum()*0.1033:.2f} s  resid sd={np.std(yc[m]-pred):.2f} W  (sd y={yc[m].std():.1f})")
    print(" NNLS h (k=0..):", np.round(h, 3))
    print(" OLS  h (k=0..):", np.round(hu, 3))
    # compare with simple alternatives
    for L in range(3, 9):
        r = np.std(yc[m] - Xc[m][:, L])
        print(f"   point sample at lag {L}: resid sd {r:.2f}", end=";")
    print()
    for a, b in [(1, 5), (2, 6), (3, 7), (2, 7), (1, 6), (3, 8), (4, 8), (1, 10), (2, 11), (0, 9)]:
        r = np.std(yc[m] - Xc[m][:, a:b + 1].mean(1))
        print(f"   boxcar lags {a}-{b}: resid sd {r:.2f}", end=";")
    print()
np.save(os.path.join(A, "bms_fir_all.npy"), nnls(Xc, yc)[0])
# gain check between the two sensors on segment means (static relation)
seg = df.groupby(["boot", "date", "seg"]).agg(P=("P_bms", "mean"), Pm=("P_mb", "mean")).reset_index()
for b, g in seg.groupby("boot"):
    if len(g) < 5: continue
    sl, ic = np.polyfit(g.Pm, g.P, 1)
    print(f"boot {b} {g.date.iloc[0]}: P_bms = {sl:.3f} * P_mb + {ic:.1f}   (n={len(g)}, P_mb range {g.Pm.min():.0f}-{g.Pm.max():.0f})")
