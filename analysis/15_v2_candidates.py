"""Candidate (a) for model v2, judged by held-out data: tie the copper coefficients of joints that share the same
   actuator (fewer, better identified parameters). Several tying schemes are compared with v1.
   Candidates (b) dual-sensor target and (c) accumulated-tau^2 creep are in 16_v2_dual_sensor_thermal.py;
   the temperature-rise factor that v2 finally uses is tested in 18_innovation_checks.py (F4) and 17_finalize_v2.py."""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1power as g
ROOT = sys.argv[1]; A = os.path.join(ROOT, "analysis")
pd.set_option("display.width", 250, "display.max_rows", 400, "display.max_columns", 60)
df = g.load(ROOT)
KW = dict(fixed={"mech_pos": 1.0}, upper={"mech_neg": 1.0})
X = g.features(df, dict(cu="type", mech="split")); X.insert(0, "const", 1.0)
meta_all, Xf_all = g.bms_align(df, X); k = (meta_all.fsm != 1).values
meta, Xf = meta_all[k], Xf_all[k]; y = meta.P_bms.values

# ------------------------------------------------------------------ (a) coefficient tying
# Actuator assignment of the 29-DoF G1 as used in Unitree's own simulation configs:
#   N7520-14.3: hip_pitch, hip_yaw, waist_yaw | N7520-22.5: hip_roll, knee | N5020-16: ankles, waist roll/pitch, shoulder, elbow, wrist_roll
#   W4010-25: wrist_pitch, wrist_yaw. Ankle and waist roll/pitch are two-motor parallel mechanisms, so their joint-space coefficients are kept separate.
TIES = {
 "T16 (v1: one per joint type)": {t: [t] for t in g.TYPES},
 "T10 (v1 with one coefficient for the whole arm)": {**{t: [t] for t in g.TYPES[:9]}, "arm": g.TYPES[9:]},
 "T8 (by actuator model)": {"N7520_14": ["hip_pitch", "hip_yaw", "waist_yaw"], "N7520_22": ["hip_roll", "knee"], "ankle_pitch": ["ankle_pitch"],
                            "ankle_roll": ["ankle_roll"], "waist_roll": ["waist_roll"], "waist_pitch": ["waist_pitch"],
                            "N5020_arm": ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll"], "W4010": ["wrist_pitch", "wrist_yaw"]},
 "T9 (T8, hip_pitch/hip_yaw apart from waist_yaw)": {"hip_py": ["hip_pitch", "hip_yaw"], "waist_yaw": ["waist_yaw"], "N7520_22": ["hip_roll", "knee"], "ankle_pitch": ["ankle_pitch"],
                            "ankle_roll": ["ankle_roll"], "waist_roll": ["waist_roll"], "waist_pitch": ["waist_pitch"],
                            "N5020_arm": ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll"], "W4010": ["wrist_pitch", "wrist_yaw"]},
 "T7 (T8, wrists joined to the arm)": {"N7520_14": ["hip_pitch", "hip_yaw", "waist_yaw"], "N7520_22": ["hip_roll", "knee"], "ankle_pitch": ["ankle_pitch"],
                            "ankle_roll": ["ankle_roll"], "waist_roll": ["waist_roll"], "waist_pitch": ["waist_pitch"], "arm": g.TYPES[9:]},
 "T6 (T8, ankle/waist roll tied, ankle/waist pitch tied)": {"N7520_14": ["hip_pitch", "hip_yaw", "waist_yaw"], "N7520_22": ["hip_roll", "knee"], "par_pitch": ["ankle_pitch", "waist_pitch"],
                            "par_roll": ["ankle_roll", "waist_roll"], "N5020_arm": ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll"], "W4010": ["wrist_pitch", "wrist_yaw"]},
}
def tie(Xsrc, groups):
    out = pd.DataFrame({"const": Xsrc["const"]})
    for name, types in groups.items(): out[f"cu_{name}"] = Xsrc[[f"cu_{t}" for t in types]].sum(1)
    out["mech_pos"] = Xsrc["mech_pos"]; out["mech_neg"] = Xsrc["mech_neg"]
    return out

def seg_err(pred, mask=None):
    e = pd.Series(pred - y, index=meta.index)
    if mask is not None: e = e[mask]
    return e.groupby(meta.segid[e.index].values).mean()

rows, lao = [], {}
acts = sorted(meta.action.unique())
isA = (meta.robot == "A").values; loco = meta.action.isin(g.LOCO).values
for name, groups in TIES.items():
    Xt = tie(Xf, groups)
    p, C = g.lobo_cv(meta, Xt, y, meta.boot.values, **KW)
    s = g.summarize(meta, p, y, name)
    r = dict(model=name, k=Xt.shape[1] - 1, lobo_rmse=s["rmse"], lobo_seg_mae=s["seg_mae"], lobo_seg_med=seg_err(p).abs().median(), lobo_seg_p90=s["seg_p90"], lobo_seg_max=s["seg_max"],
             boot_min=s["boot_min"], boot_max=s["boot_max"])
    # leave one action out
    pa, _ = g.lobo_cv(meta, Xt, y, meta.action.values, **KW)
    sa = seg_err(pa); act_of = meta.groupby("segid").action.first().reindex(sa.index)
    lao[name] = sa.groupby(act_of.values).mean()
    r["lao_seg_mae"] = sa.abs().mean(); r["lao_seg_max"] = sa.abs().max(); r["lao_reject_bias"] = lao[name]["reject"]
    r["lao_worst_action_bias"] = lao[name].abs().max()
    # robot B -> robot A
    c = g.fit_bounded(Xt[~isA], y[~isA], **KW); pA = Xt.values @ c.values
    sA = seg_err(pA, isA); r["B_to_A_bias"] = sA.mean(); r["B_to_A_max"] = sA.abs().max()
    # no locomotion data -> locomotion
    c = g.fit_bounded(Xt[~loco], y[~loco], **KW); pL = Xt.values @ c.values
    sL = seg_err(pL, loco); r["noloco_to_loco_mae"] = sL.abs().mean(); r["noloco_to_loco_max"] = sL.abs().max()
    # no arm-gesture data (stand + locomotion + squat only) -> gestures
    gest = ~meta.action.isin(g.LOCO + ["stand still", "squat"]).values
    c = g.fit_bounded(Xt[~gest], y[~gest], **KW); pG = Xt.values @ c.values
    sG = seg_err(pG, gest); r["nogesture_to_gesture_mae"] = sG.abs().mean(); r["nogesture_to_gesture_max"] = sG.abs().max()
    rows.append(r)
    call = g.fit_bounded(Xt, y, **KW)
    print(f"\n{name}: all-data coefficients"); print(call.round(4).to_string())
R = pd.DataFrame(rows).set_index("model")
print("\n== (a) coefficient tying: held-out errors (W) =="); print(R.round(2).T.to_string())
print("\nleave-one-action-out, mean segment bias by held-out action (W):"); print(pd.DataFrame(lao).round(1).to_string())
R.round(3).to_csv(os.path.join(A, "v2_tying_comparison.csv"), encoding="utf-8-sig")
pd.DataFrame(lao).round(2).to_csv(os.path.join(A, "v2_tying_leave_action_out.csv"), encoding="utf-8-sig")
