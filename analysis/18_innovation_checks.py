"""Quick feasibility checks for possible paper contributions, all on the existing data:
   F1  physics-structured model vs black-box regressors under distribution shift;
   F2  motor temperature dynamics driven by the model's copper loss (lumped thermal model);
   F3  what determines standing power (stance / posture)."""
import sys, os, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1power as g
ROOT = sys.argv[1]; A = os.path.join(ROOT, "analysis")
pd.set_option("display.width", 250, "display.max_rows", 400, "display.max_columns", 60)
df = g.load(ROOT)
M2 = json.load(open(os.path.join(ROOT, "g1_power_model_v2.json"), encoding="utf-8"))
KW = dict(fixed={"mech_pos": 1.0}, upper={"mech_neg": 1.0})
tau, dq, T = g.arrays(df)
bj = np.array([M2["b"][n.replace("left_", "").replace("right_", "")] for n in g.JN])
which = set(sys.argv[2:]) or {"F1", "F2", "F3"}

# ------------------------------------------------------------------ F1
if "F1" in which:
    from sklearn.linear_model import RidgeCV
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X16 = g.features(df, dict(cu="type", mech="split")); X16.insert(0, "const", 1.0)
    X16T = g.features(df, g.V2_SPEC); X16T.insert(0, "const", 1.0)
    pm = tau * dq
    si = df.groupby("segid", sort=False).cumcount().values
    Tst = pd.DataFrame(np.where((si < 15)[:, None], T, np.nan)).groupby(df.segid.values).transform("mean").values
    # black-box inputs: the same signals the physics model sees, per joint: tau^2, tau*dq, |dq| and the temperature rise since segment start
    XJ = pd.DataFrame(np.column_stack([tau ** 2, pm, np.abs(dq), T - Tst]), index=df.index,
                      columns=[f"t2_{j}" for j in range(29)] + [f"pm_{j}" for j in range(29)] + [f"adq_{j}" for j in range(29)] + [f"dT_{j}" for j in range(29)])
    meta_all, Xall = g.bms_align(df, pd.concat([X16, X16T.add_suffix("@T"), XJ], axis=1)); k = (meta_all.fsm != 1).values
    meta, Xa = meta_all[k], Xall[k]; y = meta.P_bms.values; segid = meta.segid.values
    Xv1 = Xa[X16.columns]; Xtied = g.tie_types(Xv1); Xbb = Xa[XJ.columns].values
    Xv2 = g.tie_types(Xa[[c + "@T" for c in X16.columns]].rename(columns=lambda c: c[:-2]))
    act_of = meta.groupby("segid").action.first()
    def phys(Xsrc):
        return lambda tr, te: Xsrc[te].values @ g.fit_bounded(Xsrc[tr], y[tr], **KW).values
    def sk(make):
        def f(tr, te):
            mdl = make(); mdl.fit(Xbb[tr], y[tr]); return mdl.predict(Xbb[te])
        return f
    MODELS = {
        "physics v2 (10 parameters)": phys(Xv2), "physics, tied coefficients only": phys(Xtied), "physics v1 (18 parameters)": phys(Xv1),
        "ridge, 116 per-joint features": sk(lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))),
        "random forest": sk(lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=0)),
        "gradient boosting": sk(lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=4, random_state=0)),
        "MLP 64-64": sk(lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=600, early_stopping=True, random_state=0))),
    }
    gest = ~meta.action.isin(g.LOCO + ["stand still", "squat"]).values; loco = meta.action.isin(g.LOCO).values; isA = (meta.robot == "A").values
    def cv(f, groups):
        p = np.zeros(len(y))
        for b in sorted(pd.unique(groups)):
            te = groups == b; p[te] = f(~te, te)
        return p
    def segstats(e, mask=None):
        s = pd.Series(e if mask is None else e[mask]).groupby(segid if mask is None else segid[mask]).mean()
        return s
    rows = []
    for name, f in MODELS.items():
        t0 = time.time()
        e1 = segstats(cv(f, meta.boot.values) - y)
        pa = cv(f, meta.action.values); e2 = segstats(pa - y); by = e2.groupby(act_of.reindex(e2.index).values).mean()
        p3 = np.zeros(len(y)); p3[gest] = f(~gest, gest); e3 = segstats(p3 - y, gest)
        p4 = np.zeros(len(y)); p4[loco] = f(~loco, loco); e4 = segstats(p4 - y, loco)
        p5 = np.zeros(len(y)); p5[isA] = f(~isA, isA); e5 = segstats(p5 - y, isA)
        rows.append(dict(model=name, session_mae=e1.abs().mean(), session_max=e1.abs().max(), action_mae=e2.abs().mean(), action_max=e2.abs().max(),
                         worst_action=f"{by.abs().idxmax()} {by[by.abs().idxmax()]:+.1f}", nogest_mae=e3.abs().mean(), nogest_max=e3.abs().max(),
                         noloco_mae=e4.abs().mean(), noloco_max=e4.abs().max(), BtoA_mae=e5.abs().mean(), BtoA_max=e5.abs().max(), sec=time.time() - t0))
        print(rows[-1], flush=True)
    R = pd.DataFrame(rows).set_index("model")
    print("\n== F1. segment-mean errors on held-out data (W): session / action class / no gestures->gestures / no locomotion->locomotion / robot B->A =="); print(R.round(2).to_string())
    R.round(3).to_csv(os.path.join(A, "f1_physics_vs_blackbox.csv"), encoding="utf-8-sig")

# ------------------------------------------------------------------ F2
if "F2" in which:
    print("\n== F2. motor temperature rate within a segment vs copper loss of that joint (v2 coefficients) ==")
    rec = []
    for sid, idx in df.groupby("segid", sort=False).indices.items():
        if len(idx) < 120: continue
        t = df.t.values[idx]; a, b = idx[:20], idx[-20:]
        D = t[-20:].mean() - t[:20].mean()
        for j in range(29):
            rec.append(dict(segid=sid, boot=df.boot.values[idx[0]], action=df.action.values[idx[0]], j=j, type=g.JN[j].replace("left_", "").replace("right_", ""),
                            dT=T[b, j].mean() - T[a, j].mean(), D=D, t2=(tau[idx, j] ** 2).mean(), Tm=T[idx, j].mean(), T0=T[a, j].mean()))
    S = pd.DataFrame(rec); S["rate"] = S.dT / S.D; S["Pcu"] = S.t2 * S.type.map(M2["b"])
    grp_of = {t: gn for gn, types in M2["groups"].items() for t in types}
    S["grp"] = S.type.map(grp_of)
    # Ankle and waist roll/pitch are each driven by two motors through a linkage: tau/q are reported in joint space (pitch, roll),
    # but the two temperatures belong to the two physical motors. Give each motor half of the pair's copper loss.
    for a_, b_, nm in [(4, 5, "ankle (2 motors)"), (10, 11, "ankle (2 motors)"), (13, 14, "waist roll/pitch (2 motors)")]:
        pa = S[S.j == a_].set_index("segid").Pcu; pb = S[S.j == b_].set_index("segid").Pcu
        half = (pa + pb) / 2
        for jj in (a_, b_):
            mj = S.j == jj; S.loc[mj, "Pcu"] = half.reindex(S.loc[mj, "segid"]).values; S.loc[mj, "grp"] = nm
    print("temperature change within a segment (K): overall sd %.2f; share of joint-segments with |dT| >= 1 K: %.1f%%" % (S.dT.std(), 100 * (S.dT.abs() >= 1).mean()))
    S.round(5).to_csv(os.path.join(A, "f2_thermal_joint_segments.csv"), index=False, encoding="utf-8-sig")
    out = []
    rng = np.random.default_rng(0)
    for gn, x in S.groupby("grp"):
        Xd = np.column_stack([x.Pcu, x.Tm, np.ones(len(x))]); yv = x.rate.values
        def fit(Xd, yv):
            c, *_ = np.linalg.lstsq(Xd, yv, rcond=None); return c
        c = fit(Xd, yv); r2 = 1 - ((yv - Xd @ c) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
        segs = x.segid.unique(); pos = {s: np.where(x.segid.values == s)[0] for s in segs}; bs = []
        for r in range(200):
            ii = np.concatenate([pos[s] for s in rng.choice(segs, len(segs))]); bs.append(fit(Xd[ii], yv[ii]))
        bs = np.array(bs)
        a_, k_ = c[0], -c[1]
        out.append(dict(group=gn, n=len(x), r2=r2, heat_K_per_J=a_, heat_lo=np.quantile(bs[:, 0], .025), heat_hi=np.quantile(bs[:, 0], .975),
                        C_th_J_per_K=1 / a_ if a_ > 0 else np.nan, cool_per_s=k_, cool_lo=np.quantile(-bs[:, 1], .025), cool_hi=np.quantile(-bs[:, 1], .975),
                        tau_th_s=1 / k_ if k_ > 0 else np.nan, T_env_eff=c[2] / k_ if k_ > 0 else np.nan, R_th_K_per_W=a_ / k_ if k_ > 0 else np.nan,
                        Pcu_max=x.Pcu.max(), T_max=x.Tm.max()))
    O = pd.DataFrame(out).set_index("group"); print(O.round(4).to_string())
    O.round(5).to_csv(os.path.join(A, "f2_thermal_fit.csv"), encoding="utf-8-sig")
    # same fit with the heating coefficient only (no cooling term), restricted to segments where the motor carries a clear load
    print("heating rate per joule, joint-segments with copper loss > 3 W, rate = a * Pcu + c:")
    for gn, x in S[S.Pcu > 3].groupby("grp"):
        if len(x) < 15: continue
        c = np.polyfit(x.Pcu, x.rate, 1); print("  %-28s a = %.4f K/J  (C_th %.1f J/K)  r = %.2f  n = %d" % (gn, c[0], 1 / c[0], np.corrcoef(x.Pcu, x.rate)[0, 1], len(x)))
    # steady-state view for the ankle motors
    al = S[S.grp == "ankle (2 motors)"]; c = np.polyfit(al.Pcu, al.Tm, 1)
    print("ankle motors, all segments: temperature level vs copper loss per motor: slope %.2f K/W, intercept %.1f C, r = %.2f, n = %d, Pcu %.1f-%.1f W, T %.0f-%.0f C"
          % (c[0], c[1], np.corrcoef(al.Pcu, al.Tm)[0, 1], len(al), al.Pcu.min(), al.Pcu.max(), al.Tm.min(), al.Tm.max()))
    pb = al.groupby("boot").agg(T=("Tm", "mean"), Tmax=("Tm", "max"), Pcu=("Pcu", "mean")).round(1)
    print("ankle motors per power-on session (mean temperature, max, mean copper loss per motor):"); print(pb.T.to_string())
    # worked example: an arm hold
    rj = S[(S.action == "reject") & (S.grp == "N5020_arm")].sort_values("Pcu", ascending=False).groupby("segid").head(1)
    print("reject, most loaded arm joint per segment: copper loss %.1f W (mean), measured rise %.1f K over %.0f s -> %.2f K/s" % (rj.Pcu.mean(), rj.dT.mean(), rj.D.mean(), (rj.dT / rj.D).mean()))

# ------------------------------------------------------------------ F3
if "F3" in which:
    print("\n== F3. standing: what determines the power? (quiet frames: sum|dq| < 0.5 rad/s, non-damping, 2 s blocks) ==")
    quiet = (np.abs(dq).sum(1) < 0.5) & (df.fsm != 1).values
    d = df[quiet].copy(); iq = np.where(quiet)[0]
    d["legcu"] = (bj[g.LEG] * tau[iq][:, g.LEG] ** 2).sum(1); d["armcu"] = (bj[g.ARM] * tau[iq][:, g.ARM] ** 2).sum(1); d["waistcu"] = (bj[g.WAIST] * tau[iq][:, g.WAIST] ** 2).sum(1)
    for nm, js in [("tau_ankle_pitch", [4, 10]), ("tau_knee", [3, 9]), ("tau_hip_pitch", [0, 6]), ("tau_hip_roll", [1, 7]), ("tau_ankle_roll", [5, 11])]:
        d[nm] = tau[iq][:, js].mean(1)
    for nm, js in [("q_ankle_pitch", [4, 10]), ("q_knee", [3, 9]), ("q_hip_pitch", [0, 6])]:
        d[nm] = df[[f"q{j}" for j in js]].values[iq].mean(1)
    d["imu_pitch"] = d.imu_rpy1; d["imu_roll"] = d.imu_rpy0
    d["blk"] = (d.groupby("segid").cumcount() // 20)
    cols = ["P_bms", "legcu", "armcu", "waistcu", "tau_ankle_pitch", "tau_knee", "tau_hip_pitch", "tau_hip_roll", "tau_ankle_roll", "q_ankle_pitch", "q_knee", "q_hip_pitch", "imu_pitch", "imu_roll"]
    B = d.groupby(["segid", "blk"]).agg(n=("t", "size"), boot=("boot", "first"), action=("action", "first"), **{c: (c, "mean") for c in cols}).reset_index()
    B = B[B.n >= 15]
    low = B[B.armcu < 3.0]                       # arms down: plain standing
    print("blocks: %d (arms-down standing: %d, from %d segments, %d boots)" % (len(B), len(low), low.segid.nunique(), low.boot.nunique()))
    print("arms-down standing: BMS power %.1f-%.1f W (sd %.1f); leg copper loss %.1f-%.1f W (sd %.1f)" % (low.P_bms.min(), low.P_bms.max(), low.P_bms.std(), low.legcu.min(), low.legcu.max(), low.legcu.std()))
    print("correlation with leg copper loss (arms-down standing blocks):")
    print(low[cols].corr()["legcu"].round(2).to_string())
    lq = low.dropna(subset=["q_ankle_pitch", "imu_pitch"])
    def r2fit(cs):
        Xd = np.column_stack([lq[c] for c in cs] + [np.ones(len(lq))]); c, *_ = np.linalg.lstsq(Xd, lq.legcu.values, rcond=None)
        return 1 - ((lq.legcu - Xd @ c) ** 2).sum() / ((lq.legcu - lq.legcu.mean()) ** 2).sum(), c
    for cs in [["imu_pitch"], ["q_ankle_pitch"], ["q_knee"], ["q_ankle_pitch", "q_knee", "q_hip_pitch"], ["q_ankle_pitch", "q_knee", "q_hip_pitch", "imu_pitch", "imu_roll"], ["tau_ankle_pitch"]]:
        r2, c = r2fit(cs); print("  leg copper loss ~ %-55s R2 = %.2f  (n = %d blocks)" % (" + ".join(cs), r2, len(lq)))
    # quadratic in ankle torque: how much of standing power is the ankle?
    ank = (bj[4] * (tau[iq][:, [4, 10]] ** 2).sum(1) + bj[5] * (tau[iq][:, [5, 11]] ** 2).sum(1)); d["ankcu"] = ank
    lo2 = d[d.armcu < 3.0]
    print("arms-down standing frames: share of leg copper loss that is in the ankles: %.0f%% (mean ankle %.1f W of leg %.1f W); knee+hip %.1f W" % (100 * lo2.ankcu.sum() / lo2.legcu.sum(), lo2.ankcu.mean(), lo2.legcu.mean(), (lo2.legcu - lo2.ankcu).mean()))
    per = low.groupby("segid").agg(boot=("boot", "first"), P=("P_bms", "mean"), legcu=("legcu", "mean"), tau_ankle_pitch=("tau_ankle_pitch", "mean"), q_ankle_pitch=("q_ankle_pitch", "mean"), q_knee=("q_knee", "mean"), imu_pitch=("imu_pitch", "mean"))
    print("per power-on session, arms-down standing (mean over segments):"); print(per.groupby("boot").mean().round(3).to_string())
    per.round(4).to_csv(os.path.join(A, "f3_standing_blocks.csv"), encoding="utf-8-sig")
    lowf = d[d.armcu < 3.0]
    print("arms-down standing frames by session and controller state (fsm, mode_machine):")
    print(lowf.groupby(["boot", "fsm", "mode_machine"]).agg(n=("t", "size"), P=("P_bms", "mean"), q_ankle_pitch=("q_ankle_pitch", "mean"), q_knee=("q_knee", "mean"),
                                                             tau_ankle_pitch=("tau_ankle_pitch", "mean"), legcu=("legcu", "mean")).round(3).to_string())
    two = low.dropna(subset=["q_ankle_pitch"]).assign(stance=lambda x: np.where(x.q_ankle_pitch < -0.2, "deep (ankle < -0.20 rad)", "shallow"))
    print(two.groupby("stance").agg(blocks=("P_bms", "size"), segments=("segid", "nunique"), boots=("boot", "nunique"), P=("P_bms", "mean"), P_sd=("P_bms", "std"), legcu=("legcu", "mean"),
                                    q_ankle=("q_ankle_pitch", "mean"), q_knee=("q_knee", "mean"), q_hip=("q_hip_pitch", "mean"), tau_ankle=("tau_ankle_pitch", "mean")).round(3).to_string())

# ------------------------------------------------------------------ F4: copper loss with the textbook temperature coefficient of copper
if "F4" in which:
    print("\n== F4. copper loss scaled by (1 + alpha * (T - Tref)) with the reported motor temperature; alpha is not fitted ==")
    seg0 = df.groupby("segid", sort=False).cumcount().values
    T0 = pd.DataFrame(np.where((seg0 < 15)[:, None], T, np.nan)).groupby(df.segid.values).transform("mean").values
    def feats(alpha, mode):
        ref = 45.0 if mode == "level" else T0
        t2 = tau ** 2 * (1.0 + alpha * (T - ref))
        cols = {"const": np.ones(len(df))}
        for t, idx in g.TYPE_IDX.items(): cols[f"cu_{t}"] = t2[:, idx].sum(1)
        pm = tau * dq; cols["mech_pos"] = np.clip(pm, 0, None).sum(1); cols["mech_neg"] = np.clip(pm, None, 0).sum(1)
        return g.tie_types(pd.DataFrame(cols, index=df.index))
    rows = []
    for mode in ["level", "rise within segment"]:
        for alpha in [0.0, 0.002, 0.0039, 0.006]:
            if mode != "level" and alpha == 0: continue
            ma, Xa_ = g.bms_align(df, feats(alpha, mode)); kq = (ma.fsm != 1).values; ma, Xa_ = ma[kq], Xa_[kq]; yv = ma.P_bms.values
            p, _ = g.lobo_cv(ma, Xa_, yv, ma.boot.values, **KW); s = g.summarize(ma, p, yv)
            pa, _ = g.lobo_cv(ma, Xa_, yv, ma.action.values, **KW); ea = pd.Series(pa - yv).groupby(ma.segid.values).mean()
            hm = ma.arm_holding.fillna(False).astype(bool).values & (ma.action == "reject").values; sl = []
            for sgi, x in pd.DataFrame({"t": ma.t.values[hm], "r": (yv - p)[hm]}).groupby(ma.segid.values[hm]):
                if len(x) >= 12: sl.append(np.polyfit(x.t - x.t.iloc[0], x.r, 1)[0] * 10)
            hm2 = ma.arm_holding.fillna(False).astype(bool).values & ~(ma.action == "reject").values; sl2 = []
            for sgi, x in pd.DataFrame({"t": ma.t.values[hm2], "r": (yv - p)[hm2]}).groupby(ma.segid.values[hm2]):
                if len(x) >= 12: sl2.append(np.polyfit(x.t - x.t.iloc[0], x.r, 1)[0] * 10)
            rows.append(dict(reference=mode, alpha=alpha, session_seg_mae=s["seg_mae"], session_seg_max=s["seg_max"], session_bias_min=s["boot_min"], session_bias_max=s["boot_max"], rmse=s["rmse"],
                             action_seg_mae=ea.abs().mean(), action_seg_max=ea.abs().max(), reject_hold_slope_W_per_10s=np.mean(sl), n_reject=len(sl), other_hold_slope_W_per_10s=np.mean(sl2), n_other=len(sl2)))
    R4 = pd.DataFrame(rows); print(R4.round(3).to_string(index=False))
    R4.round(4).to_csv(os.path.join(A, "f4_temperature_coefficient.csv"), index=False, encoding="utf-8-sig")
