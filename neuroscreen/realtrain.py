"""
Train and persist the real-person wearable screening model (Paper 2 pathway).

Trains a gradient-boosting classifier on the wearable feature CSV (real IEEE
DataPort data when present, otherwise the bundled synthetic frame) and saves a
self-describing artifact that ``neuroscreen.detect`` loads to screen a real
subject.

Run:
    python -m neuroscreen.realtrain                    # synthetic fallback
    python -m neuroscreen.realtrain --data data/paper2.csv   # real data
"""

from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from . import dataset as ds
from .models import confusion, per_class_metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")
ARTIFACT = os.path.join(MODEL_DIR, "realmodel.joblib")


def _load(data_path):
    if data_path:
        X, y, names, report = ds.load_csv(data_path)
        source = f"real:{os.path.basename(data_path)}"
    else:
        df = ds.synthetic_frame()
        names = [c for c in ds.CANONICAL if c in df.columns]
        X = df[names].to_numpy(float)
        y = df["label"].to_numpy()
        report = {"n_subjects": len(y), "features_present": names,
                  "class_counts": {c: int(np.sum(y == c)) for c in ds.CLASSES}}
        source = "synthetic"
        print(f"No --data given: training on synthetic wearable frame "
              f"({len(y)} subjects). Provide the IEEE DataPort CSV with --data "
              f"to train on real patients.")
    return X, y, names, report, source


def _cv_eval(X, y, labels):
    """Leave-one-subject-out where feasible, else stratified 5-fold."""
    counts = {c: int(np.sum(y == c)) for c in labels}
    use_loso = len(y) <= 160 and min(counts.values()) >= 2
    splitter = (LeaveOneOut() if use_loso
                else StratifiedKFold(5, shuffle=True, random_state=0))
    preds = np.empty(len(y), dtype=object)
    for tr, te in splitter.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.08, max_depth=3,
            subsample=0.9, random_state=0).fit(sc.transform(X[tr]), y[tr])
        preds[te] = clf.predict(sc.transform(X[te]))
    acc, rows = per_class_metrics(y, preds, labels)
    scheme = "leave-one-subject-out" if use_loso else "stratified 5-fold"
    return acc, rows, confusion(y, preds, labels).tolist(), scheme


def main():
    ap = argparse.ArgumentParser(description="Train the real wearable screening model.")
    ap.add_argument("--data", help="path to wearable feature CSV (IEEE DataPort)")
    args = ap.parse_args()

    labels = ds.CLASSES
    X, y, names, report, source = _load(args.data)
    present = [c for c in labels if np.any(y == c)]
    print(f"\nTraining data: {len(y)} subjects x {len(names)} features "
          f"({source}); classes present: {present}")

    print("Cross-validating...")
    acc, rows, cm, scheme = _cv_eval(X, y, present)
    print(f"  {scheme} accuracy = {acc*100:.1f}%")
    for c in present:
        r = rows[c]
        print(f"    {c:<8} sens={r['sensitivity']*100:4.0f}%  "
              f"spec={r['specificity']*100:4.0f}%  (n={r['support']})")

    # Final model on all data.
    scaler = StandardScaler().fit(X)
    clf = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.08, max_depth=3,
        subsample=0.9, random_state=0).fit(scaler.transform(X), y)

    artifact = {
        "version": 1,
        "source": source,
        "feature_names": names,
        "classes": list(clf.classes_),
        "scaler": scaler,
        "clf": clf,
        "train_median": np.median(X, axis=0),           # impute missing at inference
        "class_means": {n: {c: float(np.mean(X[y == c, j]))
                            for c in present}
                        for j, n in enumerate(names)},
        "cohort_mean": X.mean(0).tolist(),
        "cohort_std": (X.std(0) + 1e-9).tolist(),
        "importances": sorted(
            ({"feature": n, "importance": float(v)}
             for n, v in zip(names, clf.feature_importances_)),
            key=lambda d: d["importance"], reverse=True),
        "cv": {"scheme": scheme, "accuracy": acc, "per_class": rows,
               "confusion": cm, "labels": present},
        "report": report,
    }
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(artifact, ARTIFACT)
    with open(os.path.join(MODEL_DIR, "realmodel_meta.json"), "w") as f:
        json.dump({k: artifact[k] for k in
                   ("version", "source", "feature_names", "classes", "cv", "report")},
                  f, indent=2, default=str)
    print(f"\nSaved model artifact -> {os.path.relpath(ARTIFACT, ROOT)}")
    print("Screen a real subject with:  python -m neuroscreen.detect --input subject.csv")


if __name__ == "__main__":
    main()
