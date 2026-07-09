"""
Screen a real person for peripheral / autonomic neuropathy.

Takes a CSV of one or more subjects' wearable features (gait + plantar pressure
+ HRV, as captured by IMUs, a pressure insole and an ECG/HRV strap), aligns the
columns to the trained model's schema, and prints a screening report with the
predicted class, confidence, a clinical risk level, the markers that most drove
the decision, and a recommendation.

Usage:
    python -m neuroscreen.detect --template subject_template.csv   # make a blank input
    python -m neuroscreen.detect --input subject.csv               # screen subject(s)
    python -m neuroscreen.detect --input subject.csv --json        # machine-readable
"""

from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd

from . import dataset as ds
from .realtrain import ARTIFACT

RISK = {
    "Healthy": ("LOW", "No neuropathy indicators. Routine diabetic follow-up."),
    "DPN": ("HIGH", "Peripheral-neuropathy pattern. Refer for nerve-conduction "
                    "study / monofilament testing and foot-care review."),
    "CAN": ("HIGH", "Autonomic (cardiac) pattern. Refer for cardiovascular "
                    "autonomic reflex testing; review orthostatic symptoms."),
}


def load_model():
    if not os.path.exists(ARTIFACT):
        raise SystemExit(
            "No trained model found. Run:  python -m neuroscreen.realtrain "
            "[--data data/paper2.csv]  first.")
    return joblib.load(ARTIFACT)


def build_matrix(df: pd.DataFrame, model):
    """Align an input dataframe to the model's feature order, mapping column
    synonyms and imputing any missing markers with training medians."""
    colmap = ds.map_columns(df.columns)          # source -> canonical
    inv = {v: k for k, v in colmap.items()}
    names = model["feature_names"]
    med = np.asarray(model["train_median"], float)
    X = np.empty((len(df), len(names)), float)
    used, imputed = [], []
    for j, key in enumerate(names):
        if key in inv:
            col = pd.to_numeric(df[inv[key]], errors="coerce").to_numpy(float)
            col = np.where(np.isnan(col), med[j], col)
            X[:, j] = col
            used.append(key)
        else:
            X[:, j] = med[j]
            imputed.append(key)
    return X, used, imputed


def explain(x_row, model, pred, k=5):
    """Top markers pushing this subject toward the predicted class, expressed as
    deviation from the Healthy cohort mean, weighted by model importance."""
    names = model["feature_names"]
    cmeans = model["class_means"]
    std = np.asarray(model["cohort_std"], float)
    imp = {d["feature"]: d["importance"] for d in model["importances"]}
    healthy = "Healthy" if "Healthy" in model["classes"] else model["classes"][0]
    out = []
    for j, n in enumerate(names):
        base = cmeans.get(n, {}).get(healthy, model["cohort_mean"][j])
        z = (x_row[j] - base) / std[j]
        out.append({
            "feature": n, "value": float(x_row[j]),
            "healthy_ref": float(base),
            "z": float(z), "weight": float(imp.get(n, 0.0)),
            "score": float(z * imp.get(n, 0.0)),
            "direction": "elevated" if z > 0 else "reduced",
        })
    out.sort(key=lambda d: abs(d["score"]), reverse=True)
    return out[:k]


def screen_one(x_row, model):
    Xs = model["scaler"].transform(x_row.reshape(1, -1))
    proba = model["clf"].predict_proba(Xs)[0]
    classes = list(model["clf"].classes_)
    order = np.argsort(proba)[::-1]
    pred = classes[order[0]]
    return {
        "prediction": pred,
        "confidence": float(proba[order[0]]),
        "probabilities": {classes[i]: float(proba[i]) for i in order},
        "risk_level": RISK.get(pred, ("-", ""))[0],
        "recommendation": RISK.get(pred, ("-", ""))[1],
        "top_factors": explain(x_row, model, pred),
    }


def _print_report(subj_id, res, imputed):
    p = res["prediction"]
    bar = "=" * 62
    print(f"\n{bar}\n  SUBJECT: {subj_id}")
    print(f"  SCREENING RESULT:  {p}   ({res['confidence']*100:.0f}% confidence)")
    print(f"  RISK LEVEL:        {res['risk_level']}")
    print(f"  {res['recommendation']}")
    print("  Class probabilities:")
    for c, pr in res["probabilities"].items():
        blocks = "#" * int(round(pr * 30))
        print(f"     {c:<8} {pr*100:5.1f}%  {blocks}")
    print("  Most influential markers (vs healthy reference):")
    for f in res["top_factors"]:
        print(f"     {f['feature']:<22} {f['value']:8.2f}  "
              f"({f['direction']}, z={f['z']:+.2f})")
    if imputed:
        print(f"  NOTE: {len(imputed)} marker(s) missing from input, imputed "
              f"with cohort median: {', '.join(imputed[:6])}"
              f"{'...' if len(imputed) > 6 else ''}")
    print(bar)


def main():
    ap = argparse.ArgumentParser(description="Screen a real person for neuropathy.")
    ap.add_argument("--input", help="CSV of subject wearable features")
    ap.add_argument("--template", metavar="PATH",
                    help="write a blank input template CSV and exit")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    args = ap.parse_args()

    if args.template:
        ds.write_template(args.template)
        print(f"Wrote input template -> {args.template}")
        print("Fill one row per subject with the captured feature values, then run:")
        print(f"  python -m neuroscreen.detect --input {args.template}")
        return

    if not args.input:
        ap.error("provide --input <csv> or --template <csv>")

    model = load_model()
    df = pd.read_csv(args.input)
    id_col = next((c for c in df.columns if c.lower() in ("id", "subject", "subject_id")), None)
    X, used, imputed = build_matrix(df, model)

    results = []
    for i in range(len(df)):
        res = screen_one(X[i], model)
        sid = str(df[id_col].iloc[i]) if id_col else f"row{i+1}"
        results.append({"id": sid, **res})
        if not args.json:
            _print_report(sid, res, imputed)

    if args.json:
        print(json.dumps({"model_source": model["source"],
                          "features_used": used, "features_imputed": imputed,
                          "results": results}, indent=2))
    elif model["source"] == "synthetic":
        print("\n[!] Model trained on SYNTHETIC data -- results are a demonstration, "
              "not a clinical assessment. Retrain with the IEEE DataPort CSV via "
              "`python -m neuroscreen.realtrain --data <csv>` for real screening.")


if __name__ == "__main__":
    main()
