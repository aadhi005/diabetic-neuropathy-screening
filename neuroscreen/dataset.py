"""
Real-world dataset ingestion for the wearable (Paper 2) screening pathway.

Paper 2's dataset (IEEE DataPort, DOI 10.21227/f4jr-k711) is a *feature-level*
CSV of gait / plantar-pressure / HRV markers labelled Healthy / DPN / CAN.
It is behind an IEEE DataPort subscription, so it cannot be auto-downloaded;
place the extracted CSV under ``data/`` and point the loader at it.

Because the exact column names in a third-party or freshly-captured CSV are not
known in advance, the loader is deliberately *tolerant*: it maps whatever
columns are present onto a canonical schema using a synonym table, reports what
it found and what is missing, and trains on the intersection that is available.
This makes the same code work for the bundled synthetic frame today and the
real CSV the moment it is dropped in.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Canonical wearable feature schema (Paper 2 modalities)
# ---------------------------------------------------------------------------
# Ordered, grouped by modality. Every model / detector speaks this vocabulary.
CANONICAL = [
    # --- gait (IMU-derived) ---
    "gait_cadence", "gait_stride_time", "gait_swing_ratio", "gait_double_support",
    "gait_shank_rom", "gait_thigh_rom", "gait_mean_lin_accel", "gait_mean_ang_vel",
    "gait_vertical_accel_peak",
    # --- plantar pressure (kPa), 3 regions x {peak, mean} ---
    "press_forefoot_peak", "press_forefoot_mean", "press_midfoot_peak",
    "press_midfoot_mean", "press_heel_peak", "press_heel_mean",
    # --- heart-rate variability ---
    "hrv_mean_rr", "hrv_sdrr", "hrv_rmssd", "hrv_sd1", "hrv_sd2", "hrv_sd1_sd2",
    "hrv_lf", "hrv_hf", "hrv_lfhf",
]

GROUP = {"gait": "Gait", "press": "Plantar pressure", "hrv": "Heart-rate variability"}


def group_of(name: str) -> str:
    return name.split("_", 1)[0]


# Accepted source-column synonyms -> canonical key. Matching is done on a
# normalised form (lowercased, non-alphanumerics stripped) so "SDRR (ms)",
# "sdrr_ms" and "SDRR" all collapse to the same token.
SYNONYMS = {
    "gait_cadence": ["cadence", "gaitcadence", "steprate", "stepspermin"],
    "gait_stride_time": ["stridetime", "stride", "steptime", "gaitstridetime"],
    "gait_swing_ratio": ["swingratio", "swing", "swingphase", "swingpct"],
    "gait_double_support": ["doublesupport", "doublesupportratio", "ds", "doublestance"],
    "gait_shank_rom": ["shankrom", "shankrange", "shankrotation", "shankangle", "shankpitchrom"],
    "gait_thigh_rom": ["thighrom", "thighrange", "thighrotation", "thighangle", "thighpitchrom"],
    "gait_mean_lin_accel": ["meanlinearacceleration", "linearacceleration", "meanlinaccel", "linaccel"],
    "gait_mean_ang_vel": ["meanangularvelocity", "angularvelocity", "meanangvel", "angvel"],
    "gait_vertical_accel_peak": ["verticalacceleration", "verticalaccel", "vertaccel", "verticalaccelpeak", "maxverticalaccel"],
    "press_forefoot_peak": ["forefootpeak", "forefootmax", "maxforefoot", "toespeak", "toesmax", "forefootpressuremax"],
    "press_forefoot_mean": ["forefootmean", "forefootavg", "avgforefoot", "toesmean", "forefootpressureavg"],
    "press_midfoot_peak": ["midfootpeak", "midfootmax", "maxmidfoot", "midfootpressuremax"],
    "press_midfoot_mean": ["midfootmean", "midfootavg", "avgmidfoot", "midfootpressureavg"],
    "press_heel_peak": ["heelpeak", "heelmax", "maxheel", "heelpressuremax"],
    "press_heel_mean": ["heelmean", "heelavg", "avgheel", "heelpressureavg"],
    "hrv_mean_rr": ["meanrr", "rrmean", "meanrrinterval", "avnn", "meannn"],
    "hrv_sdrr": ["sdrr", "sdnn", "stdrr"],
    "hrv_rmssd": ["rmssd"],
    "hrv_sd1": ["sd1", "poincaresd1"],
    "hrv_sd2": ["sd2", "poincaresd2"],
    "hrv_sd1_sd2": ["sd1sd2", "sd1sd2ratio", "sd1overd2", "sd1divsd2"],
    "hrv_lf": ["lf", "lowfrequency", "lfpower", "peaklowfrequency"],
    "hrv_hf": ["hf", "highfrequency", "hfpower", "peakhighfrequency"],
    "hrv_lfhf": ["lfhf", "lfhfratio", "lfoverhf"],
}

LABEL_COLUMNS = ["label", "class", "target", "diagnosis", "group", "condition", "y"]

# Normalise many spellings of the three classes to canonical labels.
LABEL_MAP = {
    "healthy": "Healthy", "control": "Healthy", "normal": "Healthy", "h": "Healthy", "0": "Healthy",
    "dpn": "DPN", "diabetic": "DPN", "diabeticperipheralneuropathy": "DPN",
    "peripheral": "DPN", "diabeticfoot": "DPN", "1": "DPN",
    "can": "CAN", "cardiac": "CAN", "cardiacautonomicneuropathy": "CAN",
    "autonomic": "CAN", "2": "CAN",
}
CLASSES = ["Healthy", "DPN", "CAN"]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


# Real exports almost always carry units in the header -- "RMSSD (ms)",
# "LF (ms^2)", "Forefoot peak [kPa]". Plain normalisation turns those into
# "rmssdms" / "lfms2", which match nothing, so the column would be silently
# dropped. Strip bracketed groups and trailing unit tokens before matching.
_BRACKETED = re.compile(r"[\(\[\{][^\)\]\}]*[\)\]\}]")
_UNIT_TOKENS = {
    "ms", "msec", "s", "sec", "hz", "kpa", "pa", "mmhg", "bpm",
    "deg", "degree", "degrees", "rad", "percent", "pct",
    "mm", "cm", "m", "g", "n", "ms2", "msec2", "au", "ratio", "mean", "avg",
}


def _norm_no_units(s: str) -> str:
    s = _BRACKETED.sub(" ", str(s).lower())
    parts = [p for p in re.split(r"[^a-z0-9]+", s) if p and p not in _UNIT_TOKENS]
    return "".join(parts)


def _build_reverse_synonyms():
    rev = {}
    for canon, syns in SYNONYMS.items():
        rev[_norm(canon)] = canon
        for s in syns:
            rev[_norm(s)] = canon
    return rev


_REV = _build_reverse_synonyms()


def map_columns(columns):
    """Return {source_column: canonical_key} for columns we recognise."""
    mapping = {}
    for col in columns:
        # Exact normalised match first; fall back to a unit-stripped match so
        # "RMSSD (ms)" still resolves, without letting units create collisions.
        key = _REV.get(_norm(col)) or _REV.get(_norm_no_units(col))
        if key and key not in mapping.values():
            mapping[col] = key
    return mapping


def find_label_column(df: pd.DataFrame):
    for c in df.columns:
        if _norm(c) in {_norm(x) for x in LABEL_COLUMNS}:
            return c
    # Fallback: a column whose values look like class names.
    for c in df.columns:
        vals = {_norm(v) for v in df[c].astype(str).unique()[:10]}
        if vals and vals <= set(LABEL_MAP):
            return c
    return None


def normalise_labels(series: pd.Series):
    return series.astype(str).map(lambda v: LABEL_MAP.get(_norm(v), None))


def load_csv(path: str, verbose: bool = True):
    """Load a wearable feature CSV into (X, y, feature_names, report).

    X: float ndarray (n_subjects x n_features_present)
    y: ndarray of canonical labels (Healthy/DPN/CAN)
    feature_names: canonical keys present, in CANONICAL order
    report: dict describing the mapping (for logging / UI)
    """
    df = pd.read_csv(path)
    colmap = map_columns(df.columns)
    label_col = find_label_column(df)
    if label_col is None:
        raise ValueError(
            f"No label column found. Expected one of {LABEL_COLUMNS} or a column "
            f"with values like {list(CLASSES)}. Columns seen: {list(df.columns)}")

    present = [k for k in CANONICAL if k in colmap.values()]
    if not present:
        raise ValueError(
            "None of the canonical wearable features were recognised in the CSV. "
            f"Columns seen: {list(df.columns)}. See dataset.CANONICAL / SYNONYMS.")

    inv = {v: k for k, v in colmap.items()}          # canonical -> source col
    X = df[[inv[k] for k in present]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    # Drop any recognised-but-empty columns (all NaN) so they can't poison training.
    keep = ~np.all(np.isnan(X), axis=0)
    present = [k for k, ok in zip(present, keep) if ok]
    X = X[:, keep]
    y = normalise_labels(df[label_col]).to_numpy()

    # Drop rows with an unmappable label or all-NaN features.
    ok = np.array([lab in CLASSES for lab in y]) & ~np.all(np.isnan(X), axis=1)
    X, y = X[ok], y[ok]
    # Impute remaining NaNs with column medians.
    med = np.nanmedian(X, axis=0)
    X = np.where(np.isnan(X), med, X)

    missing = [k for k in CANONICAL if k not in present]
    report = {
        "path": path, "n_subjects": int(len(y)), "label_column": label_col,
        "features_present": present, "features_missing": missing,
        "class_counts": {c: int(np.sum(y == c)) for c in CLASSES},
    }
    if verbose:
        print(f"Loaded {report['n_subjects']} subjects from {path}")
        print(f"  label column: '{label_col}'  ->  {report['class_counts']}")
        print(f"  features recognised: {len(present)}/{len(CANONICAL)}")
        if missing:
            print(f"  not found (ok, ignored): {', '.join(missing)}")
    return X, y, present, report


def write_template(path: str, example: bool = True):
    """Write an empty CSV with the canonical columns (+ one example row) so a
    real recording can be formatted correctly for the detector."""
    cols = CANONICAL + ["label"]
    rows = []
    if example:
        rows.append({**{k: "" for k in CANONICAL}, "label": "Healthy"})
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
    return path


def synthetic_frame(seed: int = 11):
    """Build a wearable-only DataFrame from the synthetic generator so the real
    pipeline is fully testable before the IEEE dataset is available.

    Labels use Paper 2's space: NN/AN/SN map to Healthy vs DPN by symptom, and
    CAN stays CAN. (SN/AN carry peripheral involvement -> DPN; NN -> Healthy.)
    """
    from .signals import build_cohort, generate_wearable  # noqa: F401
    cohort = build_cohort(n_per_class=36, seed=seed)
    peripheral_to_paper2 = {"NN": "Healthy", "AN": "DPN", "SN": "DPN", "CAN": "CAN"}
    rows, labels = [], []
    for s in cohort:
        w = s["wearable"]
        # Only emit markers the synthetic generator actually produces; the
        # detector imputes any canonical marker a model was not trained on.
        rows.append({k: w[k] for k in CANONICAL if k in w})
        labels.append(peripheral_to_paper2[s["label"]])
    present = [k for k in CANONICAL if k in rows[0]]
    df = pd.DataFrame(rows, columns=present)
    df["label"] = labels
    return df


if __name__ == "__main__":
    df = synthetic_frame()
    print("synthetic wearable frame:", df.shape)
    print(df["label"].value_counts().to_dict())
