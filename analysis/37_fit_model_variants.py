"""Rebuild the three alternative parameter sets in Tables 3 and 12.

From the package root run analysis/01_load.py . and analysis/03_sessions_offsets.py .
with Python, then run python analysis/37_fit_model_variants.py .
"""
import argparse
import json
from pathlib import Path

import pandas as pd

import g1power as g


BOUNDS = {"fixed": {"mech_pos": 1.0}, "upper": {"mech_neg": 1.0}}
GEAR_RATIO_SQUARED = (22.5 / 14.3) ** 2


def segment_errors(residuals, segments):
    errors = pd.Series(residuals).groupby(segments.to_numpy()).mean().abs()
    return {"segments": len(errors), "mae_W": float(errors.mean()), "max_W": float(errors.max())}


def held_out_errors(meta, design, power):
    scores = {}
    for name, column in (("session", "boot"), ("action_class", "action")):
        pred, _ = g.lobo_cv(meta, design, power, meta[column].to_numpy(), **BOUNDS)
        scores[name] = segment_errors(pred - power, meta.segid)
    locomotion = meta.action.isin(g.LOCO)
    gestures = ~meta.action.isin(g.LOCO + ["stand still", "squat"])
    for name, test in (("no_gestures_to_gestures", gestures), ("no_locomotion_to_locomotion", locomotion)):
        coef = g.fit_bounded(design.loc[~test], power[~test], **BOUNDS)
        residuals = design.loc[test].to_numpy() @ coef.to_numpy() - power[test]
        scores[name] = segment_errors(residuals, meta.loc[test, "segid"])
    return scores


def fit_variant(meta, aligned, name, alpha, ratio, ratio_source, primary):
    design = aligned.copy()
    if ratio is not None:
        # b14 = ratio * b22, so both groups are fitted by a single coefficient.
        design["cu_N7520_22"] += ratio * design.pop("cu_N7520_14")
    power = meta.P_bms.to_numpy()
    coef = g.fit_bounded(design, power, **BOUNDS)
    if ratio is not None:
        coef["cu_N7520_14"] = ratio * coef["cu_N7520_22"]
    group_coef = {group: float(coef[f"cu_{group}"]) for group in g.ACT_GROUPS}
    joint_coef = {
        joint_type: group_coef[group]
        for group, types in g.ACT_GROUPS.items()
        for joint_type in types
    }
    return {
        "version": name,
        "name": "G1 whole-body BMS power model, " + name,
        "equation": primary["equation"],
        "units": primary["units"],
        "joint_order": g.JN,
        "P0": float(coef["const"]),
        "mech_pos_coef": float(coef["mech_pos"]),
        "eta_regen": float(coef["mech_neg"]),
        "alpha_cu": alpha,
        "temp_reference": primary["temp_reference"] if alpha else None,
        "fixed_ratio_b_N7520_14_over_22": ratio,
        "ratio_source": ratio_source,
        "groups": g.ACT_GROUPS,
        "b_group": group_coef,
        "b": joint_coef,
        "bms_response_fir_10hz": g.BMS_FIR.tolist(),
        # The telemetry envelope is common to these fits on the same source data.
        "envelope": primary["envelope"],
        "fitted_on": {"bms_samples": len(meta), "segments": meta.segid.nunique(), "sessions": meta.boot.nunique()},
        "confidence": {},
        "uncertainty_note": "Variant-specific confidence and prediction intervals were not estimated.",
        "held_out_segment_mean_errors": held_out_errors(meta, design, power),
    }


def main(root):
    frames = g.load(root)
    with (root / "g1_power_model_v2.json").open(encoding="utf-8") as stream:
        primary = json.load(stream)
    output = root / "model_variants"
    output.mkdir(exist_ok=True)
    variants = [
        ("n7520_ratio_1p55", g.ALPHA_CU, 1.55, "Ratio from the heating-rate regression"),
        ("n7520_ratio_2p48", g.ALPHA_CU, GEAR_RATIO_SQUARED, "(22.5 / 14.3)^2; displayed as 2.48"),
        ("no_temperature", 0.0, None, None),
    ]
    for name, alpha, ratio, source in variants:
        features = g.features(frames, {**g.V2_SPEC, "alpha": alpha})
        features.insert(0, "const", 1.0)
        meta, aligned = g.bms_align(frames, features)
        meta = meta.loc[meta.fsm != 1]
        aligned = g.tie_types(aligned.loc[meta.index])
        model = fit_variant(meta, aligned, name, alpha, ratio, source, primary)
        path = output / f"g1_power_model_{name}.json"
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(model, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(path.name)
        for test, score in model["held_out_segment_mean_errors"].items():
            print(f"  {test}: MAE {score['mae_W']:.3f} W, maximum {score['max_W']:.3f} W")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parents[1])
    main(parser.parse_args().root.resolve())
