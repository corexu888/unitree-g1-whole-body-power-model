"""Unitree G1 whole-robot electrical power model (battery pack power as reported by the BMS).

    P [W] = P0
          + sum over groups  b_group * sum_{j in group} tau_j^2 * k_j   (copper loss)
          + sum_j max(tau_j * dq_j, 0)                                   (mechanical output, coefficient fixed to 1)
          + eta_regen * sum_j min(tau_j * dq_j, 0)                       (braking, partly returned to the bus)
    k_j = 1 + 0.0039 * (T_j - T_j at the start of the window)           (v2 only, and only when temperatures are given)

Inputs are the 29 `tau_est` [N*m] and `dq` [rad/s] values of rt/lowstate, in SDK joint order, and optionally the
29 motor `temperature` values [deg C].
Two coefficient files live in the same folder:
    g1_power_model_v2.json (default)  one copper coefficient per actuator model (8 in total) + temperature-rise factor
    g1_power_model.json    (v1)       one copper coefficient per joint type (16 in total)
Only numpy is required.

Usage as a library:
    from g1_power_model import G1PowerModel
    m = G1PowerModel()                     # v2; G1PowerModel(version="v1") for the first version
    p = m.predict(tau, dq)                 # tau, dq: arrays of shape (N, 29) or (29,)  -> W, instantaneous
    p = m.predict(tau, dq, temp=T)         # T: (N, 29) motor temperatures; reference = mean of the first 15 rows,
                                           #    or pass temp_ref=(29,) taken while the robot stood with its arms down
    mean, lo, hi = m.mean_power_ci(tau, dq, temp=T)   # mean power of the window with a 95% interval (v2 only)
    dp = m.predict_net(tau, dq, baseline=slice(0, 20))   # power relative to a baseline window (the published arm model's framing)
    warnings = m.check_domain(tau, dq)     # list of strings; empty when inside the tested range

Usage on a telemetry file recorded by the test logger (compares with the BMS reading):
    python g1_power_model.py [--v1] "data/.../unitree_robot_telemetry.jsonl" [more files ...]

Validity: see 功率模型报告.md. In short - trust averages over >= 2 s (standing / arm gestures) or >= 10 s
(walking); do not use the instantaneous value for peak power; absolute level is tied to the BMS, which was
never checked against an external meter.
"""
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_FILES = {"v1": "g1_power_model.json", "v2": "g1_power_model_v2.json"}


class G1PowerModel:
    def __init__(self, path=None, version="v2"):
        with open(path or os.path.join(_HERE, _FILES[version]), encoding="utf-8") as fh:
            self.spec = json.load(fh)
        self.version = self.spec.get("version", "v1")
        self.joints = self.spec["joint_order"]
        self.P0 = self.spec["P0"]
        self.eta_regen = self.spec["eta_regen"]
        self.types = list(self.spec["b"].keys())
        strip = lambda n: n.replace("left_", "").replace("right_", "")
        self.type_of = [strip(n) for n in self.joints]
        self.b = np.array([self.spec["b"][t] for t in self.type_of])            # per-joint copper coefficient
        self.fir = np.array(self.spec["bms_response_fir_10hz"])
        self.env = self.spec["envelope"]["per_joint_type"]
        self.conf = self.spec["confidence"]
        self.groups = self.spec.get("groups")                                    # v2: {group: [joint types]}
        self.alpha = self.spec.get("alpha_cu", 0.0)                              # v2: copper temperature coefficient, 1/K

    @staticmethod
    def _check(tau, dq):
        tau = np.atleast_2d(np.asarray(tau, float)); dq = np.atleast_2d(np.asarray(dq, float))
        if tau.shape[1] != 29 or dq.shape != tau.shape:
            raise ValueError("tau and dq must have shape (N, 29) in SDK joint order")
        return tau, dq

    def _tau2(self, tau, temp=None, temp_ref=None):
        """tau^2 per joint, scaled for the rise of the motor temperature above its reference (v2, when temp is given)."""
        t2 = tau ** 2
        if temp is None or not self.alpha:
            return t2
        temp = np.atleast_2d(np.asarray(temp, float))
        if temp.shape != tau.shape:
            raise ValueError("temp must have the same shape as tau")
        ref = temp[:15].mean(0) if temp_ref is None else np.asarray(temp_ref, float)
        return t2 * (1.0 + self.alpha * (temp - ref))

    # ------------------------------------------------------------------ prediction
    def components(self, tau, dq, temp=None, temp_ref=None):
        """Returns dict of arrays (W): P0, copper (per joint, shape (N,29)), mech_pos, mech_neg(regen, <=0)."""
        tau, dq = self._check(tau, dq)
        pm = tau * dq
        return {"P0": np.full(len(tau), self.P0), "copper": self.b * self._tau2(tau, temp, temp_ref),
                "mech_pos": np.clip(pm, 0, None).sum(1), "mech_neg": self.eta_regen * np.clip(pm, None, 0).sum(1)}

    def predict(self, tau, dq, temp=None, temp_ref=None):
        """Instantaneous whole-robot electrical power estimate [W] for each sample."""
        c = self.components(tau, dq, temp, temp_ref)
        return c["P0"] + c["copper"].sum(1) + c["mech_pos"] + c["mech_neg"]

    def predict_bms_like(self, tau, dq, temp=None, temp_ref=None):
        """What the BMS would display for ~10 Hz samples: the BMS value behaves like a ~0.6 s moving average
        of the true power, delayed by ~0.4 s. The first 8 samples are NaN. Only valid for ~10 Hz input."""
        p = self.predict(tau, dq, temp, temp_ref)
        out = np.convolve(p, self.fir, mode="full")[:len(p)]
        out[:len(self.fir) - 1] = np.nan
        return out

    def predict_net(self, tau, dq, baseline, temp=None, temp_ref=None):
        """Power relative to the mean over a baseline window (index array or slice), e.g. standing still before a motion.
        This is the quantity modelled by the published arm-only model; here it includes waist and leg reactions."""
        p = self.predict(tau, dq, temp, temp_ref)
        return p - p[baseline].mean()

    def mean_power_ci(self, tau, dq, level=0.95, temp=None, temp_ref=None):
        """Mean power of the whole window and an interval for it: (mean, lo, hi) in W. v2 only.
        The spread combines parameter uncertainty (bootstrap coefficient sets), session-to-session and
        segment-to-segment scatter seen in held-out tests. It widens for loading patterns that rely on weakly
        identified coefficients, but it does not guarantee coverage for motions unlike anything in the fitting data."""
        u = self.spec.get("uncertainty")
        if not u or not self.groups:
            raise ValueError("mean_power_ci needs the v2 coefficient file")
        tau, dq = self._check(tau, dq)
        pm = tau * dq
        t2 = self._tau2(tau, temp, temp_ref).mean(0)
        x = [1.0] + [sum(t2[j] for j, tt in enumerate(self.type_of) if tt in types) for types in self.groups.values()] \
            + [np.clip(pm, None, 0).sum(1).mean()]
        S = np.asarray(u["bootstrap_samples"])                                  # columns: P0, b_<group>..., eta_regen
        mech_pos = np.clip(pm, 0, None).sum(1).mean()
        mean = float(self.predict(tau, dq, temp, temp_ref).mean())
        sd = float(np.sqrt((S @ np.asarray(x) + mech_pos).std() ** 2 + u["session_sd"] ** 2 + u["segment_sd"] ** 2))
        z = {0.68: 1.0, 0.9: 1.645, 0.95: 1.96, 0.99: 2.576}.get(level, 1.96)
        return mean, mean - z * sd, mean + z * sd

    # ------------------------------------------------------------------ domain check
    def check_domain(self, tau, dq, low_conf_watts=5.0):
        """Flags inputs outside what the fitting data covered, and cases where poorly identified terms matter."""
        tau, dq = self._check(tau, dq)
        msgs = []
        for t in self.types:
            idx = [j for j, tt in enumerate(self.type_of) if tt == t]
            tmax = np.abs(tau[:, idx]).max(); dmax = np.abs(dq[:, idx]).max()
            e = self.env[t]
            if tmax > e["tau_abs_max"]:
                msgs.append(f"{t}: |tau| up to {tmax:.1f} N*m exceeds the largest value in the fitting data ({e['tau_abs_max']:.1f})")
            elif np.percentile(np.abs(tau[:, idx]), 50) > e["tau_abs_p99"]:
                msgs.append(f"{t}: sustained |tau| above the 99th percentile of the fitting data ({e['tau_abs_p99']:.1f} N*m)")
            if dmax > e["dq_abs_max"]:
                msgs.append(f"{t}: |dq| up to {dmax:.2f} rad/s exceeds the fitting data ({e['dq_abs_max']:.2f})")
            c = self.conf.get(f"b_{t}", "")
            if c in ("low", "not identified"):
                # how much could this joint type matter? use the upper end of its confidence interval
                hi = self.spec["b_ci95"][t][1]
                w = (hi * tau[:, idx] ** 2).sum(1).mean()
                if w > low_conf_watts:
                    msgs.append(f"{t}: coefficient is '{c}'; with the upper bound of its interval this term would average {w:.1f} W here")
        return msgs


# ---------------------------------------------------------------------- CLI: compare with a recorded file
def _read_jsonl(path):
    t, tau, dq, temp, p = [], [], [], [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            mo = o["lowstate"]["motors"]
            t.append(o["time_unix"]); tau.append([m["tau_est"] for m in mo]); dq.append([m["dq"] for m in mo])
            temp.append([m["temperature"] for m in mo])
            p.append(o["bms"]["voltage_v"] * abs(o["bms"]["current_a"]))          # discharge-positive pack power
    return np.array(t), np.array(tau), np.array(dq), np.array(temp, float), np.array(p)


def main(argv):
    version = "v2"
    if argv and argv[0] in ("--v1", "--v2"):
        version = argv[0][2:]; argv = argv[1:]
    if not argv:
        print(__doc__); return
    m = G1PowerModel(version=version)
    print(f"model {m.version}: P0 = {m.P0:.1f} W, eta_regen = {m.eta_regen:.2f}")
    print(f"{'measured':>9} {'model':>8} {'diff':>7} {'95% interval':>15} {'dur_s':>6}  file")
    for path in argv:
        t, tau, dq, temp, p = _read_jsonl(path)
        if len(t) < 24:
            print(f"{'':>9} {'':>8} {'':>7} {'':>15} {t[-1] - t[0] if len(t) else 0:6.1f}  {path}  (too short)"); continue
        ref = temp[:15].mean(0)
        pb = m.predict_bms_like(tau, dq, temp, ref); ok = ~np.isnan(pb)
        ci = ""
        if m.version == "v2":
            _, lo, hi = m.mean_power_ci(tau[ok], dq[ok], temp=temp[ok], temp_ref=ref); ci = f"{lo:.1f}-{hi:.1f}"
        print(f"{p[ok].mean():9.1f} {pb[ok].mean():8.1f} {pb[ok].mean() - p[ok].mean():+7.1f} {ci:>15} {t[-1] - t[0]:6.1f}  {path}")
        for w in m.check_domain(tau, dq):
            print("      ! " + w)


if __name__ == "__main__":
    main(sys.argv[1:])
