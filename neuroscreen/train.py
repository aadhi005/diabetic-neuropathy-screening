"""
End-to-end training / evaluation pipeline.

Steps:
  1. Generate the synthetic multi-modal cohort.
  2. Extract Paper-1 COP features and fuse with Paper-2 wearable markers.
  3. Cross-validate (leave-one-subject-out) both the COP kNN ensemble and the
     fused gradient-boosting model.
  4. Train final models on all data and export a compact, browser-runnable
     classifier + metrics to ``web/model.json`` and a feature table to
     ``data/features.csv``.

Run:  python -m neuroscreen.train
"""

from __future__ import annotations

import csv
import json
import os
import time

import numpy as np
from sklearn.model_selection import LeaveOneOut

from .features import extract_cop_features
from .models import (ComponentEnsemble, DiagnosisPathway2,
                     build_gradient_boosting, confusion, per_class_metrics)
from .signals import (CLASS_NAMES, CLASSES, FS, build_cohort,
                      generate_wearable)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_feature_matrix(cohort):
    """Return (X, y, feature_names, cop_names, wear_names, raw_rows)."""
    rows, y = [], []
    cop_names = wear_names = None
    t0 = time.time()
    for i, s in enumerate(cohort):
        cop = extract_cop_features(s["ap"], s["ml"], FS)
        wear = s["wearable"]
        if cop_names is None:
            cop_names = list(cop.keys())
            wear_names = list(wear.keys())
        rows.append([cop[k] for k in cop_names] + [wear[k] for k in wear_names])
        y.append(s["label"])
        if (i + 1) % 20 == 0:
            print(f"  ...{i + 1}/{len(cohort)} subjects "
                  f"({time.time() - t0:.0f}s)")
    X = np.array(rows, float)
    return X, np.array(y), cop_names + wear_names, cop_names, wear_names, rows


def loso_eval(model_factory, X, y, labels):
    """Leave-one-subject-out cross-validation returning pooled predictions."""
    loo = LeaveOneOut()
    preds = np.empty(len(y), dtype=object)
    for tr, te in loo.split(X):
        m = model_factory()
        m.fit(X[tr], y[tr])
        preds[te[0]] = m.predict(X[te])[0]
    acc, rows = per_class_metrics(y, preds, labels)
    return acc, rows, confusion(y, preds, labels).tolist()


def main():
    print("[1/4] Generating synthetic cohort...")
    cohort = build_cohort(n_per_class=36, seed=7)
    from collections import Counter
    print("      groups:", dict(Counter(s["label"] for s in cohort)))

    print("[2/4] Extracting multi-domain features (this takes ~1-2 min)...")
    X, y, names, cop_names, wear_names, raw_rows = build_feature_matrix(cohort)
    print(f"      feature matrix: {X.shape[0]} subjects x {X.shape[1]} features")

    print("[3/4] Leave-one-subject-out cross-validation...")
    # Paper-1 posture pathway: COP kNN ensemble on the 3 severity stages only
    # (NN / AN / SN), excluding the cardiac-autonomic class that balance data
    # cannot see. Reproduces the paper's standing-balance screening task.
    posture = np.array([lab in ("NN", "AN", "SN") for lab in y])
    Xp, yp = X[posture], y[posture]
    pos_acc, pos_rows, pos_cm = loso_eval(
        lambda: ComponentEnsemble(names, k=3), Xp, yp, ["NN", "AN", "SN"])
    print(f"      Posture COP ensemble, NN/AN/SN (Paper 1) accuracy = {pos_acc*100:.1f}%")

    # Paper-1 DP-2: the two-stage cascade -- (NN+AN) vs SN, then NN vs AN.
    # The paper reports this beating the single 3-class model, which is what
    # makes early (asymptomatic) detection viable; DP-1's ~86% does not.
    dp2_acc, dp2_rows, dp2_cm = loso_eval(
        lambda: DiagnosisPathway2(names, k=3), Xp, yp, ["NN", "AN", "SN"])
    print(f"      Posture DP-2 cascade, NN/AN/SN (Paper 1) accuracy = {dp2_acc*100:.1f}%")

    # Same ensemble asked to screen ALL four classes, including CAN, to show
    # why posture alone is insufficient for autonomic involvement.
    ens_acc, ens_rows, ens_cm = loso_eval(
        lambda: ComponentEnsemble(names, k=3), X, y, CLASSES)
    print(f"      Posture COP ensemble, all 4 classes      accuracy = {ens_acc*100:.1f}%")

    # Novel fusion pathway (COP + gait + pressure + HRV) via gradient boosting.
    gb_acc, gb_rows, gb_cm = loso_eval(
        build_gradient_boosting, X, y, CLASSES)
    print(f"      Fused multi-modal GB, all 4 classes       accuracy = {gb_acc*100:.1f}%")

    print("[4/4] Training final models and exporting...")
    gb = build_gradient_boosting().fit(X, y)
    importances = sorted(
        ({"feature": n, "importance": float(v)}
         for n, v in zip(names, gb.feature_importances_)),
        key=lambda d: d["importance"], reverse=True)

    # Standardised reference set so the browser can run a live kNN vote.
    mean, std = X.mean(0), X.std(0)
    std[std == 0] = 1.0
    Xz = (X - mean) / std
    class_means = {
        n: {c: float(np.mean(X[y == c, j])) for c in CLASSES}
        for j, n in enumerate(names)}

    # A few example subjects the UI can load for demonstration.
    examples = []
    rng = np.random.default_rng(1)
    for c in CLASSES:
        idxs = np.where(y == c)[0]
        pick = int(rng.choice(idxs))
        examples.append({"label": c,
                         "features": {n: float(X[pick, j])
                                      for j, n in enumerate(names)}})

    model = {
        "classes": CLASSES,
        "class_names": CLASS_NAMES,
        "feature_names": names,
        "cop_feature_names": cop_names,
        "wearable_feature_names": wear_names,
        "scaler": {"mean": mean.tolist(), "std": std.tolist()},
        "reference": {"X": Xz.round(4).tolist(), "y": y.tolist()},
        "knn_k": 5,
        "class_means": class_means,
        "importances": importances,
        "metrics": {
            "posture": {"name": "Posture COP kNN ensemble - severity stages (Paper 1)",
                        "labels": ["NN", "AN", "SN"],
                        "accuracy": pos_acc, "per_class": pos_rows,
                        "confusion": pos_cm},
            "dp2": {"name": "Posture DP-2 two-stage cascade (Paper 1)",
                    "labels": ["NN", "AN", "SN"],
                    "accuracy": dp2_acc, "per_class": dp2_rows,
                    "confusion": dp2_cm},
            "ensemble": {"name": "Posture COP ensemble - all 4 classes",
                         "labels": CLASSES,
                         "accuracy": ens_acc, "per_class": ens_rows,
                         "confusion": ens_cm},
            "gb": {"name": "Fused multi-modal gradient boosting (novel)",
                   "labels": CLASSES,
                   "accuracy": gb_acc, "per_class": gb_rows, "confusion": gb_cm},
        },
        "examples": examples,
    }

    os.makedirs(os.path.join(ROOT, "web"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    with open(os.path.join(ROOT, "web", "model.json"), "w") as f:
        json.dump(model, f)
    # Also emit a JS global so the dashboard opens directly from file:// with
    # no local server (browsers block fetch() of local JSON).
    with open(os.path.join(ROOT, "web", "model.js"), "w") as f:
        f.write("window.MODEL = ")
        json.dump(model, f)
        f.write(";")
    with open(os.path.join(ROOT, "data", "features.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"] + names)
        for s, row in zip(cohort, raw_rows):
            w.writerow([s["id"], s["label"]] + [f"{v:.6g}" for v in row])

    print("\nDone. Exports:")
    print("  web/model.json     (browser-runnable classifier + metrics)")
    print("  data/features.csv  (full feature table)")
    print(f"\nTop discriminative features:")
    for d in importances[:8]:
        print(f"  {d['feature']:<16} {d['importance']:.3f}")


if __name__ == "__main__":
    main()
