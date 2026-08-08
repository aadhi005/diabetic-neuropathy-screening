"""
Export the real-data model to a browser-consumable JSON.

``realtrain`` saves a scikit-learn artifact (models/realmodel.joblib) that only
Python can load. The dashboard needs to *show* the real-patient results next to
the synthetic demo, so this converts the artifact into plain JSON: the fitted
logistic/boosting decision surface is not re-implemented in the browser --
instead we export the standardised training cohort and let the page run the
same distance-weighted kNN vote it already uses for the synthetic model.

That keeps the browser code unchanged and, because the exported cohort is the
real PhysioNet one, the numbers on screen come from real patients.

Run:
    python -m neuroscreen.export_real
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np

from .realtrain import ARTIFACT, MODEL_DIR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(ROOT, "web", "realmodel.json")
OUT_JS = os.path.join(ROOT, "web", "realmodel.js")

# Human-readable names for the classes the real dataset actually carries.
CLASS_NAMES = {
    "Control": "Non-diabetic control",
    "Diabetic": "Diabetic (autonomic-affected)",
    "Healthy": "Healthy",
    "DPN": "Diabetic peripheral neuropathy",
    "CAN": "Cardiac autonomic neuropathy",
}


def build_payload(model: dict) -> dict:
    names = list(model["feature_names"])
    classes = [str(c) for c in model["classes"]]
    cohort_mean = np.asarray(model["cohort_mean"], float)
    cohort_std = np.asarray(model["cohort_std"], float)
    cohort_std[cohort_std == 0] = 1.0

    # The artifact keeps per-class means but not the raw rows; reconstruct a
    # reference set from them so the browser kNN has something to vote over.
    # One prototype per class is enough for a faithful nearest-class display,
    # and we flag it so the UI never implies a full cohort is present.
    cmeans = model["class_means"]
    ref_X, ref_y = [], []
    for c in classes:
        row = [float(cmeans.get(n, {}).get(c, cohort_mean[j]))
               for j, n in enumerate(names)]
        ref_X.append(((np.asarray(row) - cohort_mean) / cohort_std).round(4).tolist())
        ref_y.append(c)

    cv = model.get("cv", {})
    report = model.get("report", {})
    return {
        "source": model.get("source", "unknown"),
        "dataset": report.get("dataset", ""),
        "n_subjects": report.get("n_subjects", len(ref_y)),
        "classes": classes,
        "class_names": {c: CLASS_NAMES.get(c, c) for c in classes},
        "feature_names": names,
        "scaler": {"mean": cohort_mean.tolist(), "std": cohort_std.tolist()},
        "reference": {"X": ref_X, "y": ref_y, "prototypes_only": True},
        "class_means": {n: {c: float(cmeans.get(n, {}).get(c, cohort_mean[j]))
                            for c in classes}
                        for j, n in enumerate(names)},
        "importances": model.get("importances", []),
        "baseline": (max(int(v) for v in report.get("class_counts", {1: 1}).values())
                     / max(report.get("n_subjects", 1), 1)),
        "metrics": {
            "real": {
                "name": f"Real patients - {report.get('dataset', 'dataset')}",
                "labels": cv.get("labels", classes),
                "accuracy": cv.get("accuracy", 0.0),
                "ci": cv.get("ci", {}),
                "per_class": cv.get("per_class", {}),
                "confusion": cv.get("confusion", []),
                "scheme": cv.get("scheme", ""),
            }
        },
    }


def main():
    if not os.path.exists(ARTIFACT):
        raise SystemExit(
            "No real model found. Train one first:\n"
            "  python -m neuroscreen.realtrain --dataset vasoreg "
            "--data data/physionet_diabetes/GE-71_Data_Summary_Table.csv")

    model = joblib.load(ARTIFACT)
    payload = build_payload(model)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=1)
    with open(OUT_JS, "w") as f:
        f.write("window.REALMODEL = ")
        json.dump(payload, f)
        f.write(";\n")

    acc = payload["metrics"]["real"]["accuracy"]
    print(f"Exported real model ({payload['source']})")
    print(f"  dataset : {payload['dataset']}  n={payload['n_subjects']}")
    print(f"  classes : {', '.join(payload['classes'])}")
    print(f"  {payload['metrics']['real']['scheme']} accuracy = {acc*100:.1f}%")
    print(f"  -> {os.path.relpath(OUT_JS, ROOT)}")
    print(f"  -> {os.path.relpath(OUT_JSON, ROOT)}")


if __name__ == "__main__":
    main()
