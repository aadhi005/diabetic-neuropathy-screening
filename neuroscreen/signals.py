"""
Synthetic multi-modal signal generation for diabetic-neuropathy screening.

Neither source paper ships its raw data locally (Paper 1's cohort is private;
Paper 2's is on IEEE DataPort), so this module synthesises a physiologically
plausible cohort whose *class-dependent behaviour* mirrors what the papers
report:

  * Center-of-Pressure (COP) sway during quiet standing  -> Paper 1
      Larger sway amplitude, higher diffusion and weaker postural control with
      increasing neuropathy severity (NN < AN < SN).
  * Gait / plantar-pressure / HRV during a short walk     -> Paper 2
      Slower cadence, higher peak plantar pressure, reduced heart-rate
      variability with peripheral / autonomic involvement.

The generator is deterministic given a seed so results are reproducible.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Cohort definition
# ---------------------------------------------------------------------------
# Paper 1 severity stages plus Paper 2 autonomic class, unified into one label
# space so a single model can screen posture *and* wearable-gait modalities.
CLASSES = ["NN", "AN", "SN", "CAN"]
CLASS_NAMES = {
    "NN": "Non-neuropathic (diabetic control)",
    "AN": "Asymptomatic peripheral neuropathy",
    "SN": "Symptomatic peripheral neuropathy",
    "CAN": "Cardiac autonomic neuropathy",
}

FS = 100.0            # COP sampling rate (Hz), as in Paper 1 (Kistler plate)
DURATION = 40.0       # seconds of quiet standing kept per trial (trimmed from 120 s)


class _ClassProfile:
    """Per-class physiological knobs used to shape every modality."""

    def __init__(self, sway, stiffness, drift, roughness,
                 cadence, peak_press, hrv_sdrr, hrv_rmssd, hrv_lfhf):
        self.sway = sway            # overall COP amplitude scale (mm)
        self.stiffness = stiffness  # postural restoring strength (higher = steadier)
        self.drift = drift          # low-frequency wandering component
        self.roughness = roughness  # high-frequency jitter / irregularity
        self.cadence = cadence      # steps/min during the walk test
        self.peak_press = peak_press  # forefoot peak plantar pressure (kPa)
        self.hrv_sdrr = hrv_sdrr    # HRV time-domain SDRR (ms)
        self.hrv_rmssd = hrv_rmssd  # HRV rMSSD (ms)
        self.hrv_lfhf = hrv_lfhf    # HRV LF/HF ratio


# Values chosen so extracted features reproduce the directional trends the
# papers describe; magnitudes are illustrative, not clinical ground truth.
PROFILES = {
    "NN":  _ClassProfile(sway=4.0,  stiffness=0.85, drift=0.20, roughness=0.35,
                         cadence=112, peak_press=210, hrv_sdrr=42, hrv_rmssd=34, hrv_lfhf=1.4),
    "AN":  _ClassProfile(sway=5.6,  stiffness=0.62, drift=0.34, roughness=0.42,
                         cadence=104, peak_press=255, hrv_sdrr=36, hrv_rmssd=27, hrv_lfhf=1.9),
    "SN":  _ClassProfile(sway=8.3,  stiffness=0.40, drift=0.55, roughness=0.55,
                         cadence=92,  peak_press=320, hrv_sdrr=30, hrv_rmssd=20, hrv_lfhf=2.6),
    "CAN": _ClassProfile(sway=5.0,  stiffness=0.66, drift=0.30, roughness=0.40,
                         cadence=100, peak_press=245, hrv_sdrr=18, hrv_rmssd=12, hrv_lfhf=3.8),
}


def _ou_sway(rng, profile, n, fs):
    """One COP component via a bounded, drift-augmented Ornstein-Uhlenbeck
    process -- a compact stand-in for the open-loop/closed-loop postural
    control model of Collins & De Luca (Paper 1, ref [28])."""
    dt = 1.0 / fs
    theta = profile.stiffness * 1.6          # mean-reversion (postural control)
    sigma = profile.sway * profile.roughness  # driving noise scale
    x = np.zeros(n)
    # Slow wandering set-point (weak long-term control -> larger with severity).
    drift = np.cumsum(rng.normal(0, profile.drift, n))
    drift = drift - np.linspace(drift[0], drift[-1], n)  # detrend endpoints
    drift *= profile.sway * 0.12
    for i in range(1, n):
        x[i] = x[i - 1] + theta * (drift[i] - x[i - 1]) * dt + sigma * np.sqrt(dt) * rng.normal()
    # Scale to a realistic mm amplitude for this class.
    x = x / (np.std(x) + 1e-9) * profile.sway
    return x


def generate_cop(rng, label, fs=FS, duration=DURATION):
    """Return (ap, ml) center-of-pressure trajectories in millimetres.

    ap  -> antero-posterior component (APc in Paper 1)
    ml  -> medial-lateral component  (MLc in Paper 1)
    The planar path formed by (ap, ml) is the statokinesigram (STKc).
    """
    profile = PROFILES[label]
    n = int(fs * duration)
    ap = _ou_sway(rng, profile, n, fs)
    # ML sway is typically smaller and, in symptomatic PPN, disproportionately
    # affected (Paper 1 discussion of the transversal component).
    ml_profile = _ClassProfile(**{**profile.__dict__})
    ml_profile.sway = profile.sway * (0.72 if label != "SN" else 0.9)
    ml = _ou_sway(rng, ml_profile, n, fs)
    return ap, ml


def generate_wearable(rng, label):
    """Return a dict of Paper-2 style gait / plantar-pressure / HRV markers."""
    p = PROFILES[label]
    def jit(v, frac):  # add subject-level variability
        return float(v * (1 + rng.normal(0, frac)))
    stride_time = 60.0 / max(jit(p.cadence, 0.06), 1e-3) * 2  # s per stride
    return {
        "gait_cadence": jit(p.cadence, 0.06),
        "gait_stride_time": stride_time,
        "gait_swing_ratio": jit(0.38 - (p.sway - 4) * 0.006, 0.05),
        "gait_double_support": jit(0.10 + (p.sway - 4) * 0.012, 0.08),
        "gait_shank_rom": jit(62 - (p.sway - 4) * 1.8, 0.07),   # deg range of motion
        "gait_thigh_rom": jit(41 - (p.sway - 4) * 1.1, 0.07),
        "press_forefoot_peak": jit(p.peak_press, 0.08),
        "press_heel_peak": jit(p.peak_press * 0.78, 0.08),
        "press_midfoot_mean": jit(p.peak_press * 0.33, 0.10),
        "hrv_sdrr": jit(p.hrv_sdrr, 0.10),
        "hrv_rmssd": jit(p.hrv_rmssd, 0.10),
        "hrv_lfhf": jit(p.hrv_lfhf, 0.12),
        "hrv_mean_rr": jit(60000.0 / (72 + (4 - p.hrv_rmssd * 0.1)), 0.05),
    }


def build_cohort(n_per_class=36, seed=7):
    """Generate a full synthetic cohort.

    Returns a list of subject dicts, each with raw COP signals and wearable
    markers plus the ground-truth label. Group sizes default to Paper 1's
    balance (36 NN / 36 AN / 32 SN); CAN is added from Paper 2.
    """
    rng = np.random.default_rng(seed)
    sizes = {"NN": n_per_class, "AN": n_per_class, "SN": max(n_per_class - 4, 8),
             "CAN": max(n_per_class // 2 - 5, 8)}
    cohort = []
    sid = 0
    for label in CLASSES:
        for _ in range(sizes[label]):
            ap, ml = generate_cop(rng, label)
            cohort.append({
                "id": f"S{sid:03d}",
                "label": label,
                "ap": ap,
                "ml": ml,
                "wearable": generate_wearable(rng, label),
            })
            sid += 1
    return cohort


if __name__ == "__main__":
    c = build_cohort()
    print(f"generated {len(c)} subjects")
    from collections import Counter
    print(Counter(s["label"] for s in c))
