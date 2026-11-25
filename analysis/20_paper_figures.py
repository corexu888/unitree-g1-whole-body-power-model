"""Draw Figures 1 through 6 using model v2, the released analysis tables and motor telemetry."""
import sys, os, json
import subprocess
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1power as g
ROOT = sys.argv[1]; A = os.path.join(ROOT, "analysis")
OUT = os.path.join(ROOT, "manuscript", sys.argv[2] if len(sys.argv) > 2 else "figures_v2"); os.makedirs(OUT, exist_ok=True)      # draft v1 used manuscript/figures

# ---- style: thin marks, hairline solid grid, text in ink colours, three validated hues at most
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.edgecolor": AXIS, "axes.linewidth": 0.7,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-", "axes.axisbelow": True, "legend.frameon": False,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 300, "pdf.fonttype": 42, "axes.titlelocation": "left"})
def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight"); fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight"); plt.close(fig)
    print("saved", name)
def panel(ax, letter, title):
    ax.set_title(f"({letter}) {title}", color=INK, pad=6)

df = g.load(ROOT)
M = json.load(open(os.path.join(ROOT, "g1_power_model_v2.json"), encoding="utf-8"))
KW = dict(fixed={"mech_pos": 1.0}, upper={"mech_neg": 1.0})
tau, dq, T = g.arrays(df)
bj = np.array([M["b"][n.replace("left_", "").replace("right_", "")] for n in g.JN])

# ------------------------------------------------------------------ Fig 1: sensor response and time series (session held out)
X16T = g.features(df, g.V2_SPEC); X16T.insert(0, "const", 1.0); Xfr = g.tie_types(X16T)
meta_all, Xf_all = g.bms_align(df, Xfr); k = (meta_all.fsm != 1).values
meta, Xf = meta_all[k], Xf_all[k]; y = meta.P_bms.values
_, C = g.lobo_cv(meta, Xf, y, meta.boot.values, **KW)
miss = sorted(set(df.segid) - set(meta.segid))
for s_ in miss:
    x_ = df[df.segid == s_]; print("segment without fitting rows:", s_, "| frames", len(x_), "| fsm values", sorted(x_.fsm.dropna().unique().tolist()), "| duration %.1f s" % (x_.t.max() - x_.t.min()))
print("fitting rows %d, segments %d, sessions %d; frames %d, segments in telemetry %d; total telemetry time %.1f min; mean frame rate %.2f Hz" % (
    len(y), meta.segid.nunique(), meta.boot.nunique(), len(df), df.segid.nunique(), df.groupby("segid").t.agg(lambda v: v.max() - v.min()).sum() / 60,
    len(df) / df.groupby("segid").t.agg(lambda v: v.max() - v.min()).sum()))
print("segment duration (s): min %.1f median %.1f max %.1f" % tuple(df.groupby("segid").t.agg(lambda v: v.max() - v.min()).quantile([0, .5, 1]).values))
print("segments per action:", meta.groupby("action").segid.nunique().to_dict())
inst = np.zeros(len(df))
for b in sorted(df.boot.unique()):
    mm = (df.boot == b).values; inst[mm] = Xfr[mm].values @ C[b].reindex(Xfr.columns).values
df["pred_inst"] = inst
K = len(g.BMS_FIR) - 1
df["pred_bms"] = df.groupby("segid").pred_inst.transform(lambda s: np.convolve(s.values, g.BMS_FIR, mode="full")[:len(s)])
df.loc[df.groupby("segid").cumcount() < K, "pred_bms"] = np.nan
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.35), gridspec_kw={"width_ratios": [0.8, 1.25, 1.25]})
lag = np.arange(len(g.BMS_FIR)) * 0.103
ax[0].bar(lag, g.BMS_FIR, width=0.07, color=BLUE)
ax[0].set_xlabel("delay (s)"); ax[0].set_ylabel("weight"); ax[0].grid(axis="x", visible=False)
cen = float((lag * g.BMS_FIR).sum()); ax[0].set_ylim(0, 0.25)
ax[0].text(0.97, 0.96, "mean delay %.2f s" % cen, transform=ax[0].transAxes, ha="right", va="top", color=INK2, fontsize=7.5)
panel(ax[0], "a", "BMS response")
for a_, (d, s, ttl, letter) in zip(ax[1:], [("2026-11-18", os.path.join("Arm", "Reject - Run 1"), "arm hold (reject)", "b"), ("2026-11-18", os.path.join("Locomotion", "Walk forward (Run 1)"), "walking, 0.5 m/s command", "c")]):
    gg = df[(df.date == d) & (df.seg == s)]; t = gg.t - gg.t.iloc[0]
    a_.plot(t, gg.pred_inst, color=AXIS, lw=0.8, label="model, instantaneous (10 Hz)", zorder=1)
    a_.plot(t, gg.P_bms, color=INK, lw=1.4, label="measured (BMS)", zorder=3)
    a_.plot(t, gg.pred_bms, color=BLUE, lw=1.4, label="model through BMS response", zorder=4)
    a_.set_xlabel("time (s)"); a_.set_ylabel("power (W)"); panel(a_, letter, ttl)
ax[2].set_ylim(100, None)
h, l = ax[1].get_legend_handles_labels(); order = [1, 2, 0]
fig.legend([h[i] for i in order], [l[i] for i in order], loc="lower center", ncol=3, bbox_to_anchor=(0.6, -0.08))
fig.tight_layout(w_pad=1.2); save(fig, "fig1_sensor_and_timeseries")
# time-resolved errors of v2 against the BMS (session held out), in short windows
fr = df.groupby("segid").cur_raw.transform(lambda s: np.r_[False, np.diff(s.values) != 0]).values.astype(bool) & df.pred_bms.notna().values & (df.fsm != 1).values
w = df[fr].copy(); w["e"] = w.pred_bms - w.P_bms; w["loco"] = w.action.isin(g.LOCO); w["i"] = df.groupby("segid").cumcount()[fr]
for sec in (2, 10):
    w["win"] = w.i // int(round(sec / 0.103))
    ww = w.groupby(["segid", "win"]).agg(e=("e", "mean"), n=("e", "size"), loco=("loco", "first")); ww = ww[ww.n >= max(2, int(sec / 0.52 * 0.6))]
    for nm, x in [("standing and gestures", ww[~ww.loco]), ("locomotion", ww[ww.loco])]:
        print("%2d s windows, %-22s n = %4d  MAE %.2f  p90 %.2f  p99 %.2f  max %.2f W" % (sec, nm, len(x), x.e.abs().mean(), x.e.abs().quantile(.9), x.e.abs().quantile(.99), x.e.abs().max()))
print("per-sample rmse (BMS rows, session held out): all %.2f, standing and gestures %.2f, locomotion %.2f W" % (np.sqrt((w.e ** 2).mean()), np.sqrt((w.e[~w.loco] ** 2).mean()), np.sqrt((w.e[w.loco] ** 2).mean())))
pk = df[(df.fsm != 1) & df.action.isin(g.LOCO) & df.P_mb.notna()].groupby("segid").agg(inst=("pred_inst", "max"), bms=("P_bms", "max"), pb=("pred_bms", "max"))
print("locomotion segments: peak of BMS reading %.0f W (mean over segments), peak of model through BMS response %.0f W, peak of instantaneous model %.0f W" % (pk.bms.mean(), pk.pb.mean(), pk.inst.mean()))

# ------------------------------------------------------------------ Fig 2: segment means, session held out
seg = pd.read_csv(os.path.join(A, "v2_segment_errors_cv.csv"), index_col=0)
seg["pred"] = seg.meas + seg.err_v2
grp = np.where(seg.robot == "A", "robot A (all actions)", np.where(seg.action.isin(g.LOCO), "robot B, locomotion", "robot B, standing and arm gestures"))
fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": [1.15, 1]})
lim = [110, 195]
ax[0].fill_between(lim, [lim[0] - 5, lim[1] - 5], [lim[0] + 5, lim[1] + 5], color=GRID, alpha=0.7, lw=0, zorder=0)
ax[0].plot(lim, lim, color=MUTED, lw=0.8, zorder=1)
for name, col, mk in [("robot B, standing and arm gestures", BLUE, "o"), ("robot B, locomotion", ORANGE, "^"), ("robot A (all actions)", AQUA, "s")]:
    x = seg[grp == name]
    short = {"robot B, standing and arm gestures": "robot B: standing, gestures", "robot B, locomotion": "robot B: locomotion", "robot A (all actions)": "robot A"}[name]
    ax[0].scatter(x.meas, x.pred, s=22, color=col, marker=mk, edgecolor="white", linewidth=0.6, label=short, zorder=3)
    print("fig2 group", name, len(x))
ax[0].set_xlim(lim); ax[0].set_ylim(lim); ax[0].set_aspect("equal")
ax[0].set_xlabel("measured segment-mean power (W)"); ax[0].set_ylabel("predicted, session held out (W)")
ax[0].legend(loc="upper left", handletextpad=0.3, borderaxespad=0.3); ax[0].text(194, 112, "shaded band: ±5 W", color=INK2, ha="right", va="bottom", fontsize=7.5)
panel(ax[0], "a", "Predicted against measured, 111 segments")
e = seg.err_v2
ax[1].hist(e, bins=np.arange(-7, 7.01, 1), color=BLUE, rwidth=0.88)
ax[1].set_xlabel("predicted − measured (W)"); ax[1].set_ylabel("segments"); ax[1].grid(axis="x", visible=False)
ax[1].text(0.03, 0.97, "median |error| %.1f W\n90%% below %.1f W\nlargest %.1f W" % (e.abs().median(), e.abs().quantile(.9), e.abs().max()), transform=ax[1].transAxes, va="top", color=INK2, fontsize=7.5)
panel(ax[1], "b", "Error of the segment mean")
fig.tight_layout(w_pad=1.5); save(fig, "fig2_heldout_session")

# ------------------------------------------------------------------ Fig 3: physics structure against black-box regressors
F1 = pd.read_csv(os.path.join(A, "f1_physics_vs_blackbox.csv"), index_col=0)
tuned_file = os.path.join(A, "f1_blackbox_nested_tuned.csv")
if os.path.exists(tuned_file):                      # black-box rows: hyper-parameters chosen inside each training fold (25_blackbox_nested_tuning.py)
    Tn = pd.read_csv(tuned_file, index_col=0).rename(index={"ridge": "ridge, 116 per-joint features"})
    for i in Tn.index:
        for c in Tn.columns:
            if c in F1.columns: F1.loc[i, c] = Tn.loc[i, c]
    print("fig3: black-box rows taken from the nested-tuned results:", list(Tn.index))
rows = [(i, lab) for i, lab in [("physics v2 (10 parameters)", "proposed (10 parameters)"), ("physics, tied coefficients only", "shared coefficients, no temperature factor (10)"),
                                ("physics v1 (18 parameters)", "per-joint-type coefficients (18)"),
                                ("ridge, 116 per-joint features", "ridge regression"), ("gradient boosting", "gradient boosting"), ("random forest", "random forest"), ("MLP 64-64", "multilayer perceptron")] if i in F1.index]
splits = [("session", "held-out session"), ("action", "held-out action class"), ("nogest", "no gestures in fit,\ntested on gestures"), ("noloco", "no locomotion in fit,\ntested on locomotion")]
fig, ax = plt.subplots(1, 4, figsize=(7.0, 2.5), sharey=True)
ypos = np.arange(len(rows))[::-1]
for a_, (key, ttl) in zip(ax, splits):
    for yy, (idx, lab) in zip(ypos, rows):
        col = BLUE if idx.startswith("physics") else MUTED
        mae, mx = F1.loc[idx, key + "_mae"], F1.loc[idx, key + "_max"]
        a_.plot([mae, mx], [yy, yy], color=col, lw=0.8, zorder=2)
        a_.scatter([mx], [yy], s=20, facecolor="white", edgecolor=col, linewidth=1.0, zorder=3)
        a_.scatter([mae], [yy], s=24, color=col, edgecolor="white", linewidth=0.6, zorder=4)
    a_.set_xscale("log"); a_.set_xlim(1, 3000); a_.set_xticks([1, 10, 100, 1000]); a_.set_xticklabels(["1", "10", r"$10^2$", r"$10^3$"])
    a_.set_title(ttl, fontsize=7.5, color=INK, loc="center", pad=5); a_.grid(axis="y", visible=False); a_.set_ylim(-0.6, len(rows) - 0.4)
ax[0].set_yticks(ypos); ax[0].set_yticklabels([lab for _, lab in rows])
for tl, (idx, _) in zip(ax[0].get_yticklabels(), rows): tl.set_color(INK if idx.startswith("physics") else INK2)
from matplotlib.lines import Line2D
fig.tight_layout(w_pad=1.0, rect=(0, 0.17, 1, 1))
fig.text(0.63, 0.115, "error of the segment-mean power (W, log scale)", ha="center", va="center", fontsize=8, color=INK2)
fig.legend([Line2D([], [], marker="o", color="none", markerfacecolor=INK2, markeredgecolor="white", markersize=5.5), Line2D([], [], marker="o", color="none", markerfacecolor="white", markeredgecolor=INK2, markersize=5)],
           ["mean absolute error", "largest error"], loc="lower center", ncol=2, bbox_to_anchor=(0.63, -0.02))
save(fig, "fig3_physics_vs_blackbox")

# ------------------------------------------------------------------ Fig 4: where the net power of a gesture goes
G = pd.read_csv(os.path.join(A, "paper_cmp_hold_by_gesture_v2.csv"), index_col=0).sort_values("meas_BMS")   # from: 14_paper_model_comparison.py . g1_power_model_v2.json
GD = pd.read_csv(os.path.join(A, "rev_gesture_decomposition.csv"), index_col=0).reindex(G.index)             # from 24_revision_analyses.py: temperature factor included
for c in ["arms", "waist", "legs"]: G[c] = GD[c]
G["kind"] = GD.kind
fig, a_ = plt.subplots(figsize=(5.2, 2.7))
yy = np.arange(len(G)); left = np.zeros(len(G))
for col, name, c in [("arms", "arms", BLUE), ("waist", "waist", ORANGE), ("legs", "legs", AQUA)]:
    v = G[col].values; pos = np.clip(v, 0, None)
    a_.barh(yy, pos, left=left, height=0.5, color=c, edgecolor="white", linewidth=1.2, label=name); left += pos
    neg = np.clip(v, None, 0)
    if (neg < 0).any(): a_.barh(yy, neg, left=0, height=0.5, color=c, edgecolor="white", linewidth=1.2)
a_.scatter(G.meas_BMS, yy, marker="D", s=26, color=INK, edgecolor="white", linewidth=0.7, zorder=5, label="measured")
a_.scatter(G.paper_coefs_arms_only, yy + 0.38, marker="v", s=24, facecolor="white", edgecolor=INK, linewidth=0.9, zorder=5, label="published arm-only model")
a_.set_yticks(yy); a_.set_yticklabels([f"{n.replace('xray', 'x-ray')} ({kd.replace('-arm', ' arm' if kd.startswith('one') else ' arms')}, n = {int(k)})" for n, k, kd in zip(G.index, G.n, G.kind)]); a_.grid(axis="y", visible=False)
a_.set_xlabel("net power during the hold, relative to standing before the motion (W)"); a_.axvline(0, color=AXIS, lw=0.7)
a_.legend(loc="lower right", ncol=1, handletextpad=0.4); a_.set_ylim(-0.6, len(G) - 0.2)
fig.tight_layout(); save(fig, "fig4_gesture_net_power")

# ------------------------------------------------------------------ Fig 5: winding temperature
subprocess.run(
    [sys.executable, os.path.join(A, "36_figure5_temperature.py"), ROOT, "--figure-dir", OUT],
    check=True,
)

# ------------------------------------------------------------------ Fig 6: standing power and stance
quiet = (np.abs(dq).sum(1) < 0.5) & (df.fsm != 1).values
armcu = (bj[g.ARM] * tau[:, g.ARM] ** 2).sum(1)
d = df[quiet & (armcu < 3.0)].copy(); iq = np.where(quiet & (armcu < 3.0))[0]
d["qa_deg"] = np.degrees(df[["q4", "q10"]].values[iq].mean(1))
ank = bj[4] * tau[iq][:, [4, 10]] ** 2 + bj[5] * tau[iq][:, [5, 11]] ** 2          # per leg: pitch + roll copper loss, shared by two motors
d["ank_motor_W"] = ank.mean(1) / 2
d["ank_T"] = T[iq][:, [4, 5, 10, 11]].mean(1)
per = d.groupby("segid").agg(n=("t", "size"), P=("P_bms", "mean"), qa=("qa_deg", "mean"), ankW=("ank_motor_W", "mean"), ankT=("ank_T", "mean"), robot=("robot", "first"))
per = per[per.n >= 20]; pq = per.dropna(subset=["qa"]).copy(); pq["stance"] = np.where(pq.qa < np.degrees(-0.2), "deep", "shallow")
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
for name, col, mk in [("shallow", BLUE, "o"), ("deep", ORANGE, "^")]:
    x = pq[pq.stance == name]
    ax[0].scatter(x.qa, x.P, s=20, color=col, marker=mk, edgecolor="white", linewidth=0.6, label=f"{name} stance (n = {len(x)})", zorder=3)
    ax[1].scatter(x.ankW, x.ankT, s=20, color=col, marker=mk, edgecolor="white", linewidth=0.6, label=f"{name} stance", zorder=3)
ax[0].set_xlabel("ankle pitch angle while standing (degrees; negative: dorsiflexion)"); ax[0].set_ylabel("whole-robot power (W)"); ax[0].legend(loc="upper right")
panel(ax[0], "a", "Standing power against stance")
ax[1].set_xlabel("modelled copper loss per ankle motor (W)"); ax[1].set_ylabel("ankle motor temperature (°C)"); ax[1].legend(loc="lower right")
panel(ax[1], "b", "Ankle motor temperature")
fig.tight_layout(w_pad=1.5); save(fig, "fig6_standing_stance")
st = pq.groupby("stance").agg(n=("P", "size"), P=("P", "mean"), P_sd=("P", "std"), qa=("qa", "mean"), ankW=("ankW", "mean"), ankT=("ankT", "mean"), ankT_max=("ankT", "max"))
print(st.round(2).to_string()); print("corr P~qa: %.3f" % np.corrcoef(pq.qa, pq.P)[0, 1], " n segments:", len(pq), " of which robot B:", int((pq.robot == "B").sum()))
st.round(3).to_csv(os.path.join(A, "paper_fig6_stance_by_segment.csv"), encoding="utf-8-sig")
