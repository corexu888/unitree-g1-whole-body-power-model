"""Shared helpers for the G1 whole-robot electrical power model."""
import os, json, re
import numpy as np, pandas as pd

JN = ['left_hip_pitch', 'left_hip_roll', 'left_hip_yaw', 'left_knee', 'left_ankle_pitch', 'left_ankle_roll',
      'right_hip_pitch', 'right_hip_roll', 'right_hip_yaw', 'right_knee', 'right_ankle_pitch', 'right_ankle_roll',
      'waist_yaw', 'waist_roll', 'waist_pitch',
      'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw', 'left_elbow', 'left_wrist_roll', 'left_wrist_pitch', 'left_wrist_yaw',
      'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw', 'right_elbow', 'right_wrist_roll', 'right_wrist_pitch', 'right_wrist_yaw']
TYPES = []                      # joint type = name without left_/right_
for n in JN:
    t = re.sub(r"^(left|right)_", "", n)
    if t not in TYPES: TYPES.append(t)
TYPE_IDX = {t: [j for j, n in enumerate(JN) if re.sub(r"^(left|right)_", "", n) == t] for t in TYPES}
LEG = list(range(0, 12)); WAIST = [12, 13, 14]; ARM = list(range(15, 29))
GROUPS = {"leg": LEG, "waist": WAIST, "arm": ARM}
LOCO = ["walk forward", "step in place", "turn around"]
ALPHA_CU = 0.0039               # temperature coefficient of copper resistance, 1/K (textbook value, not fitted)
V2_SPEC = dict(cu="type", mech="split", alpha=ALPHA_CU, Tref="segstart")
# v2: joint types driven by the same actuator model share one copper coefficient. Ankle and waist roll/pitch are two-motor
# parallel mechanisms, so their joint-space coefficients stay separate.
ACT_GROUPS = {"N7520_14": ["hip_pitch", "hip_yaw", "waist_yaw"], "N7520_22": ["hip_roll", "knee"], "ankle_pitch": ["ankle_pitch"],
              "ankle_roll": ["ankle_roll"], "waist_roll": ["waist_roll"], "waist_pitch": ["waist_pitch"],
              "N5020_arm": ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll"], "W4010": ["wrist_pitch", "wrist_yaw"]}

def tie_types(X, groups=None):
    """Sum the per-joint-type tau^2 columns (cu_<type>) of X into one column per group; other columns are kept."""
    groups = groups or ACT_GROUPS
    cu = [c for c in X.columns if c.startswith("cu_")]
    out = pd.DataFrame(index=X.index)
    if "const" in X.columns: out["const"] = X["const"]
    for name, types in groups.items(): out[f"cu_{name}"] = X[[f"cu_{t}" for t in types]].sum(1)
    for c in X.columns:
        if c not in cu and c != "const": out[c] = X[c]
    return out

# BMS reading ~ moving average of true bus power over the previous ~0.1-0.7 s (estimated in 05_bms_response.py,
# arm batches, against the 10 Hz mainboard reading; weights for frame lags 0..7, normalised to 1)
BMS_FIR = np.array([0.0, 0.062, 0.186, 0.169, 0.165, 0.175, 0.14, 0.091, 0.007]); BMS_FIR = BMS_FIR / BMS_FIR.sum()

def load(root):
    A = os.path.join(root, "analysis")
    df = pd.read_pickle(os.path.join(A, "frames.pkl")).sort_values("t").reset_index(drop=True)
    df["boot"] = pd.read_pickle(os.path.join(A, "boot.pkl")).boot.values
    df["P_bms"] = df.volt_v * df.cur_a.abs()
    df["P_mb"] = df.mb_value0 * df.mb_value2
    df["segid"] = df.date + "|" + df.seg
    return df

def arrays(df):
    tau = df[[f"tau{j}" for j in range(29)]].values.astype(float)
    dq = df[[f"dq{j}" for j in range(29)]].values.astype(float)
    T = df[[f"T{j}" for j in range(29)]].values.astype(float)
    return tau, dq, T

def features(df, spec):
    """Instantaneous regressors, one row per frame. spec: dict of options.
       cu: 'all' | 'group' | 'type' | 'joint' ; mech: None|'net'|'split' ; fric: None|'abs'|'abs+sq' ; alpha: temp coefficient (1/K), Tref"""
    tau, dq, T = arrays(df)
    alpha = spec.get("alpha", 0.0); Tref = spec.get("Tref", 50.0)
    if isinstance(Tref, str) and Tref == "segstart":         # reference = each motor's temperature over the first 1.5 s of its segment
        segi = df.groupby("segid", sort=False).cumcount().values
        Tref = pd.DataFrame(np.where((segi < 15)[:, None], T, np.nan)).groupby(df.segid.values).transform("mean").values
    t2 = tau ** 2 * (1.0 + alpha * (T - Tref))
    cols = {}
    cu = spec.get("cu", "type")
    if cu == "all": cols["cu_all"] = t2.sum(1)
    elif cu == "group":
        for g, idx in GROUPS.items(): cols[f"cu_{g}"] = t2[:, idx].sum(1)
    elif cu == "type":
        for t, idx in TYPE_IDX.items(): cols[f"cu_{t}"] = t2[:, idx].sum(1)
    elif cu == "joint":
        for j, n in enumerate(JN): cols[f"cu_{n}"] = t2[:, j]
    pm = tau * dq
    mech = spec.get("mech")
    if mech == "net": cols["mech_net"] = pm.sum(1)
    elif mech == "split":
        cols["mech_pos"] = np.clip(pm, 0, None).sum(1); cols["mech_neg"] = np.clip(pm, None, 0).sum(1)
    elif mech == "split_group":
        for g, idx in GROUPS.items():
            cols[f"mech_pos_{g}"] = np.clip(pm[:, idx], 0, None).sum(1); cols[f"mech_neg_{g}"] = np.clip(pm[:, idx], None, 0).sum(1)
    fric = spec.get("fric")
    if fric in ("abs", "abs+sq"):
        for g, idx in GROUPS.items(): cols[f"absdq_{g}"] = np.abs(dq[:, idx]).sum(1)
    if fric in ("sq", "abs+sq"):
        for g, idx in GROUPS.items(): cols[f"dq2_{g}"] = (dq[:, idx] ** 2).sum(1)
    return pd.DataFrame(cols, index=df.index)

def bms_align(df, X, fir=BMS_FIR, max_dt=0.2):
    """Apply the BMS moving-average response to the per-frame regressors inside each segment and keep only
       frames carrying a fresh BMS sample. Returns (meta, Xf) with one row per fresh BMS sample."""
    K = len(fir) - 1
    metas, xs = [], []
    Xv = X.values; names = list(X.columns)
    for sid, idx in df.groupby("segid", sort=False).indices.items():
        g = df.iloc[idx]
        t = g.t.values; I = g.cur_raw.values
        fresh = np.r_[False, np.diff(I) != 0]
        ok = np.r_[True, np.diff(t) < max_dt]
        xv = Xv[idx]
        for i in np.where(fresh)[0]:
            if i < K or not ok[i - K:i + 1].all(): continue
            w = xv[i - K:i + 1][::-1]            # lag 0..K
            xs.append(fir @ w); metas.append(idx[i])
    meta = df.iloc[metas].copy()
    Xf = pd.DataFrame(np.array(xs), columns=names, index=meta.index)
    return meta, Xf

# ---------------------------------------------------------------- fitting helpers
from scipy.optimize import lsq_linear

def fit_bounded(X, y, lower=None, upper=None, fixed=None):
    """Least squares with per-column bounds. Default: every coefficient >= 0 except 'const' (free).
       fixed: {column: value} are moved to the left-hand side."""
    fixed = fixed or {}; lower = lower or {}; upper = upper or {}
    cols = [c for c in X.columns if c not in fixed]
    yy = np.asarray(y, float).copy()
    for c, v in fixed.items():
        if c in X.columns: yy = yy - v * X[c].values
    A = X[cols].values.astype(float)
    sc = np.sqrt((A ** 2).mean(0)); sc[sc == 0] = 1
    lb = np.array([lower.get(c, -np.inf if c == "const" else 0.0) for c in cols]) * sc
    ub = np.array([upper.get(c, np.inf) for c in cols]) * sc
    r = lsq_linear(A / sc, yy, bounds=(lb, ub), method="bvls" if A.shape[1] < 60 else "trf", max_iter=5000)
    coef = pd.Series(r.x / sc, index=cols)
    for c, v in fixed.items():
        if c in X.columns: coef[c] = v
    return coef.reindex(X.columns)

def lobo_cv(meta, Xf, y, groups, **kw):
    pred = np.zeros(len(y)); coefs = {}
    for b in sorted(pd.unique(groups)):
        te = (groups == b)
        c = fit_bounded(Xf[~te], y[~te], **kw)
        pred[te] = Xf[te].values @ c.values; coefs[b] = c
    return pred, pd.DataFrame(coefs)

def summarize(meta, pred, y, label=""):
    e = pred - y
    seg = pd.Series(e).groupby(meta.segid.values).mean()
    boot = pd.Series(e).groupby(meta.boot.values).mean()
    return dict(model=label, rmse=np.sqrt(np.mean(e ** 2)), mae=np.mean(np.abs(e)), bias=e.mean(), seg_mae=seg.abs().mean(),
                seg_p90=seg.abs().quantile(.9), seg_max=seg.abs().max(), boot_min=boot.min(), boot_max=boot.max(),
                r2=1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2))
