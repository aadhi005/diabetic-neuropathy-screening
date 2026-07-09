"""
Real open-access dataset loaders.

Currently wires in the PhysioNet *Cerebral Vasoregulation in Diabetes* dataset
(open access, CC-BY, DOI 10.13026/x(see physionet.org/content/
cerebral-vasoreg-diabetes)). Its per-subject summary table gives real
non-invasive **cardiac-autonomic function** measures (Valsalva ratio, head-up
tilt HR/BP responses, orthostatic BP change) and gait speed for 28 type-2
diabetic and 22 control subjects -- exactly the autonomic axis relevant to
diabetic neuropathy, and a legitimate real-person screening task:
    Diabetic (autonomic-affected) vs Control.

The raw 3 GB signal archive is not needed: the shipped
``GE-71_Data_Summary_Table.csv`` is already feature-level.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Curated, non-invasive, non-circular feature columns (autonomic reflexes +
# gait). Deliberately excludes labs / MRI / glucose that would make the
# diabetic-vs-control task trivial or non-wearable.
VASOREG_FEATURES = [
    # Valsalva manoeuvre (cardiovagal function)
    "VM RATIO", "DBP II", "DBP IV", "VMR sys_BP", "VMR dia_BP", "VMR mn_BP",
    # Resting haemodynamics
    "HR(Baseline MR)", "sys_BP(Baseline MR)", "dia_BP(Baseline MR)", "mn_BP(Baseline MR)",
    # Chemoreflex: hypercapnia / hypocapnia HR & BP
    "HR(HyperMean)", "sys_Bp(HyperMean)", "dia_BP(HyperMean)", "mn_BP(HyperMean)",
    "HR(Hypo Mean)", "sys_BP(Hypo Mean)", "dia_BP(Hypo Mean)", "mn_BP(Hypo Mean)",
    # Head-up tilt (orthostatic autonomic response)
    "HR TILT BASELINE", "SBP TILT BASELINE1", "DBP TILT BASELINE1",
    "HR(Tilt MN)", "sys_BP(Tilt MN)", "dia_BP(Tilt MN)", "mn_BP(Tilt MN)",
    "Change Baseline To Tilt Mean BP", "Change Baseline to HypoTilt Mean BP",
    # Gait
    "SPEED(m/s)",
]

VASOREG_LABEL_COL = "group2"
VASOREG_LABEL_MAP = {"dm": "Diabetic", "dmoh": "Diabetic", "control": "Control"}
VASOREG_CLASSES = ["Control", "Diabetic"]


def load_vasoreg_diabetes(path: str, verbose: bool = True):
    """Load the PhysioNet vasoregulation-in-diabetes summary table into
    (X, y, feature_names, report)."""
    df = pd.read_csv(path)
    if VASOREG_LABEL_COL not in df.columns:
        raise ValueError(f"Expected label column '{VASOREG_LABEL_COL}' not found. "
                         f"Columns: {list(df.columns)[:12]}...")

    present = [c for c in VASOREG_FEATURES if c in df.columns]
    missing_cols = [c for c in VASOREG_FEATURES if c not in df.columns]
    X = df[present].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    y = df[VASOREG_LABEL_COL].astype(str).str.strip().str.lower().map(VASOREG_LABEL_MAP).to_numpy()

    ok = np.array([lab in VASOREG_CLASSES for lab in y]) & ~np.all(np.isnan(X), axis=1)
    X, y = X[ok], y[ok]
    med = np.nanmedian(X, axis=0)
    X = np.where(np.isnan(X), med, X)

    report = {
        "dataset": "PhysioNet Cerebral Vasoregulation in Diabetes",
        "path": path, "n_subjects": int(len(y)),
        "features_present": present, "features_missing_from_file": missing_cols,
        "class_counts": {c: int(np.sum(y == c)) for c in VASOREG_CLASSES},
        "task": "Diabetic (autonomic-affected) vs Control",
    }
    if verbose:
        print(f"Loaded {report['n_subjects']} subjects from {path}")
        print(f"  task: {report['task']}  ->  {report['class_counts']}")
        print(f"  real autonomic/gait features used: {len(present)}")
        if missing_cols:
            print(f"  columns not in file (ignored): {missing_cols}")
    return X, y, present, report


LOADERS = {"vasoreg": load_vasoreg_diabetes}
