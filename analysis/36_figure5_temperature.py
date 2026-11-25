"""Draw the manuscript's three-panel Figure 5 and export its motor-level data.

Run: python analysis/36_figure5_temperature.py .
See analysis/18_innovation_checks.py (F4) for the held-out drift in panel a.
Panels b/c use the original motor channels and the released nominal copper
coefficients. Temperature is not used to fit or adjust those coefficients.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOCOMOTION = {"walk forward", "step in place", "turn around"}
# Motor 1 is the temperature under the pitch index. Sign assignment for each
# mechanism follows the within-dataset assignment described in the manuscript.
PITCH_SIGN = {"waist": 1, "left_ankle": 1, "right_ankle": -1}
BLUE, ORANGE = "#2a78d6", "#eb6834"


def motor_type(name):
    return name.removeprefix("left_").removeprefix("right_")


def load_points(root):
    with (root / "g1_power_model_v2.json").open(encoding="utf-8") as f:
        model = json.load(f)
    joints = model["joint_order"]
    arm_names = [name for name in joints if motor_type(name) in model["groups"]["N5020_arm"]]
    index = pd.read_csv(root / "Supplementary_Table_S1.csv")
    index["segment"] = index["segment"].str.replace("\\", "/", regex=False)
    index = index.set_index(["date", "segment"])
    if not index.index.is_unique:
        raise ValueError("Duplicate date/segment keys in Supplementary Table S1")

    heating, ankles, seen = [], [], set()
    data_dir = root / "data"
    for path in sorted(data_dir.rglob("*telemetry.jsonl")):
        parts = path.relative_to(data_dir).parts
        key = parts[0][:10], "/".join(parts[1:-1])
        if key in seen:
            raise ValueError(f"Multiple telemetry logs mapped to {key}")
        seen.add(key)
        meta = index.loc[key]
        with path.open(encoding="utf-8") as f:
            frames = [json.loads(line) for line in f if line.strip()]
        if len(frames) != meta.frames:
            raise ValueError(f"Frame count disagrees with Supplementary Table S1 for {key}")
        if len(frames) < 120:
            continue
        time = np.array([frame["time_unix"] for frame in frames])
        motors = [frame["lowstate"]["motors"] for frame in frames]
        temp = np.array([[m["temperature"] for m in row] for row in motors])
        tau = np.array([[m["tau_est"] for m in row] for row in motors])
        # Use the timestamps at the centres of both 20-frame endpoint windows.
        elapsed = time[-20:].mean() - time[:20].mean()
        rate = (temp[-20:].mean(axis=0) - temp[:20].mean(axis=0)) / elapsed
        mean_temp = temp.mean(axis=0)
        mean_tau2 = (tau ** 2).mean(axis=0)
        identity = dict(segment_no=meta.no, date=key[0], segment=key[1], robot=meta.robot,
                        session=meta.session, action=meta.action, frames=len(frames))
        for name in arm_names:
            j = joints.index(name)
            loss = model["b"][motor_type(name)] * mean_tau2[j]
            heating.append(dict(**identity, group="arm", channel=name,
                                mean_temperature_C=mean_temp[j], rise_rate_K_s=rate[j],
                                equal_split_loss_W=loss, signed_cross_term_W=0.0,
                                single_motor_loss_W=loss))
        for mechanism, sign in PITCH_SIGN.items():
            pitch_name, roll_name = mechanism + "_pitch", mechanism + "_roll"
            p, r = joints.index(pitch_name), joints.index(roll_name)
            bp, br = model["b"][motor_type(pitch_name)], model["b"][motor_type(roll_name)]
            half = (bp * mean_tau2[p] + br * mean_tau2[r]) / 2
            cross = sign * np.sqrt(bp * br) * np.mean(tau[:, p] * tau[:, r])
            if mechanism == "waist":
                for j, offset in [(p, cross), (r, -cross)]:
                    heating.append(dict(**identity, group="waist", channel=joints[j],
                                        mean_temperature_C=mean_temp[j], rise_rate_K_s=rate[j],
                                        equal_split_loss_W=half, signed_cross_term_W=offset,
                                        single_motor_loss_W=half + offset))
            elif meta.robot == "B" and meta.action not in LOCOMOTION:
                ankles.append(dict(**identity, ankle=mechanism, pitch_cross_sign=sign,
                                   motor1_loss_W=half + cross, motor2_loss_W=half - cross,
                                   motor1_temperature_C=mean_temp[p], motor2_temperature_C=mean_temp[r],
                                   difference_loss_W=2 * cross,
                                   difference_temperature_K=mean_temp[p] - mean_temp[r]))
    missing = set(index.index) - seen
    if missing:
        raise ValueError(f"No telemetry log found for {len(missing)} indexed segment(s): {sorted(missing)}")
    return pd.DataFrame(heating), pd.DataFrame(ankles)


def draw_figure(drift, heating, ankles, figure_dir):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
        "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.7, "axes.labelcolor": "#52514e",
        "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.axisbelow": True,
        "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
        "legend.frameon": False, "pdf.fonttype": 42,
    })
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.9, 2.85))
    sel = drift[(drift.reference == "rise within segment") |
                ((drift.reference == "level") & (drift.alpha == 0))].sort_values("alpha")
    a.axhline(0, color="#898781", lw=0.8)
    a.axvline(3.9, color="#c3c2b7", lw=0.8)
    for key, label, color, mark in [("reject", "reject", BLUE, "o"), ("other", "other gestures", ORANGE, "^")]:
        n = int(sel["n_" + key].iloc[0])
        a.plot(sel.alpha * 1000, sel[key + "_hold_slope_W_per_10s"], color=color,
               marker=mark, ms=5, lw=1.3, mec="white", mew=0.6, label=f"{label} ({n} holds)" if key == "reject" else f"{label} ({n})")
    a.text(4.02, 5.7, "copper", fontsize=7, color="#52514e")
    a.legend(loc="lower left", handlelength=1.3, handletextpad=0.3)
    a.set(title="(a) Residual drift in static holds",
          xlabel=r"temperature coefficient in the model ($10^{-3}$ K$^{-1}$)",
          ylabel="drift of measured − model\nduring a hold (W per 10 s)")
    for group, label, color, marker in [("arm", "arm joints", BLUE, "o"), ("waist", "waist motors", ORANGE, "^")]:
        points = heating[heating.group == group]
        x, y = points.single_motor_loss_W, points.rise_rate_K_s
        b.scatter(x, y, s=9, color=color, marker=marker, alpha=0.55, linewidth=0,
                  label=f"{label}, n = {len(points)}", zorder=3)
        xx = np.array([0, x.max()])
        b.plot(xx, np.polyval(np.polyfit(x, y, 1), xx), color=color, lw=1, zorder=4)
    b.legend(loc="upper left", handlelength=1.1, handletextpad=0.3)
    b.set(title="(b) Heating rate of single motors", xlabel="modelled copper loss of the motor (W)",
          ylabel="rise rate of reported\nmotor temperature (K/s)")
    c.axvline(0, color="#898781", lw=0.65)
    c.axhline(0, color="#898781", lw=0.65)
    for side, color, marker in [("left_ankle", BLUE, "o"), ("right_ankle", ORANGE, "^")]:
        points = ankles[ankles.ankle == side]
        x, y = points.difference_loss_W, points.difference_temperature_K
        c.scatter(x, y, s=10, color=color, marker=marker, alpha=0.65, linewidth=0,
                  label=f"{side.replace('_', ' ')}, n = {len(points)}", zorder=3)
    c.legend(loc="lower right", handlelength=1.1, handletextpad=0.3)
    c.set(title="(c) The two motors of an ankle", xlabel="modelled loss, motor 1 − motor 2 (W)",
          ylabel="mean temperature,\nmotor 1 − motor 2 (K)")
    fig.tight_layout(w_pad=1.25)
    for fmt in ("png", "pdf"):
        metadata = {"CreationDate": None} if fmt == "pdf" else None
        fig.savefig(figure_dir / f"fig5_temperature.{fmt}", bbox_inches="tight", dpi=300, metadata=metadata)
    plt.close(fig)


def report_fits(heating, ankles):
    # Regression lines in the panels are unadjusted. The text additionally
    # adjusts rise rate for the mean reported motor temperature in each segment.
    for group, points in heating.groupby("group"):
        terms = {"single-motor": ["single_motor_loss_W"]}
        if group == "waist":
            terms["equal split"] = ["equal_split_loss_W"]
            terms["free cross term"] = ["equal_split_loss_W", "signed_cross_term_W"]
        for model_name, losses in terms.items():
            X = np.column_stack([np.ones(len(points)), points[losses], points.mean_temperature_C])
            y = points.rise_rate_K_s.to_numpy()
            coef = np.linalg.lstsq(X, y, rcond=None)[0]
            r2 = 1 - np.sum((y - X @ coef) ** 2) / np.sum((y - y.mean()) ** 2)
            print(f"{group}, {model_name}: n={len(y)}, heating={coef[1]:.6f} K/J, R2={r2:.6f}")
            if len(losses) == 2:
                print(f"  free cross coefficient / equal-split coefficient={coef[2] / coef[1]:.6f}")
    waist = heating[heating.group == "waist"].pivot(index="segment_no", columns="channel",
                                                values=["rise_rate_K_s", "signed_cross_term_W"])
    rate_difference = waist.rise_rate_K_s.waist_pitch - waist.rise_rate_K_s.waist_roll
    cross = waist.signed_cross_term_W.waist_pitch
    print(f"waist rate difference vs signed pitch cross term: slope={np.polyfit(cross, rate_difference, 1)[0]:.6f} K/J")
    print(f"waist rate difference vs full motor-loss difference: slope={np.polyfit(2 * cross, rate_difference, 1)[0]:.6f} K/J")
    print(f"waist rate difference vs loss difference: r={cross.corr(rate_difference):.6f}")
    for side, points in ankles.groupby("ankle"):
        x, y = points.difference_loss_W, points.difference_temperature_K
        print(f"{side}: n={len(x)}, r={x.corr(y):.6f}, slope={np.polyfit(x, y, 1)[0]:.6f} K/W")

    left = ankles[ankles.ankle == "left_ankle"]
    hot = left[left[["motor1_temperature_C", "motor2_temperature_C"]].max(axis=1) > 80]
    print(f"left ankle over 80 C: {len(hot)} segments; motor 2 over 80 C in "
          f"{(hot.motor2_temperature_C > 80).sum()} segments")
    print(f"  mean losses: motor 1={hot.motor1_loss_W.mean():.4f} W, "
          f"motor 2={hot.motor2_loss_W.mean():.4f} W, "
          f"equal split={((hot.motor1_loss_W + hot.motor2_loss_W) / 2).mean():.4f} W")
    half = (ankles.motor1_loss_W + ankles.motor2_loss_W) / 2
    temperature = np.r_[ankles.motor1_temperature_C, ankles.motor2_temperature_C]
    single = np.r_[ankles.motor1_loss_W, ankles.motor2_loss_W]
    print(f"four ankle motors: temperature vs equal-split loss r={np.corrcoef(temperature, np.r_[half, half])[0, 1]:.6f}; "
          f"vs single-motor loss r={np.corrcoef(temperature, single)[0, 1]:.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parents[1],
                        help="supplementary package root")
    parser.add_argument("--output-root", type=Path, help="write derived files below this root instead")
    parser.add_argument("--figure-dir", type=Path, help="write the PNG/PDF to this directory")
    args = parser.parse_args()
    output = args.output_root or args.root
    analysis = output / "analysis"
    figure_dir = args.figure_dir or output / "manuscript" / "figures_v2"
    drift = pd.read_csv(args.root / "analysis" / "f4_temperature_coefficient.csv")
    heating, ankles = load_points(args.root)
    analysis.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    heating.to_csv(analysis / "figure5_motor_heating.csv", index=False, float_format="%.10g", lineterminator="\n")
    ankles.to_csv(analysis / "figure5_ankle_temperature.csv", index=False, float_format="%.10g", lineterminator="\n")
    draw_figure(drift, heating, ankles, figure_dir)
    report_fits(heating, ankles)


if __name__ == "__main__":
    main()
