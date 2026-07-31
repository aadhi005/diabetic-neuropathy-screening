"""
Screen a real person for peripheral / autonomic neuropathy.

Takes a CSV of one or more subjects' wearable features (gait + plantar pressure
+ HRV, as captured by IMUs, a pressure insole and an ECG/HRV strap), aligns the
columns to the trained model's schema, and prints a screening report with the
predicted class, confidence, a clinical risk level, the markers that most drove
the decision, and a recommendation. Each screening is saved to a local SQLite
database (see ``neuroscreen.db``) so a patient's results accumulate over time.

Usage:
    python -m neuroscreen.detect --template subject_template.csv   # make a blank input
    python -m neuroscreen.detect --input subject.csv               # screen subject(s)
    python -m neuroscreen.detect --input subject.csv --json        # machine-readable
    python -m neuroscreen.detect --input subject.csv --no-save     # don't persist
    python -m neuroscreen.detect --list-patients                   # everyone on file
    python -m neuroscreen.detect --history P001                    # one patient's past results

An input CSV may optionally include identity columns -- ``patient_id`` (or
``id``/``subject``/``subject_id``), ``name``, ``age``, ``sex`` -- alongside the
feature columns; these are stored with the result but never used as features.
"""

from __future__ import annotations

import argparse
import json
import os
import re

import joblib
import numpy as np
import pandas as pd

from . import dataset as ds
from . import db as ndb
from .realtrain import ARTIFACT

RISK = {
    "Healthy": ("LOW", "No neuropathy indicators. Routine diabetic follow-up."),
    "Control": ("LOW", "No diabetic-autonomic pattern detected. Routine follow-up."),
    "DPN": ("HIGH", "Peripheral-neuropathy pattern. Refer for nerve-conduction "
                    "study / monofilament testing and foot-care review."),
    "CAN": ("HIGH", "Autonomic (cardiac) pattern. Refer for cardiovascular "
                    "autonomic reflex testing; review orthostatic symptoms."),
    "Diabetic": ("ELEVATED", "Autonomic/gait pattern consistent with diabetes-"
                             "related dysfunction. Correlate clinically; consider "
                             "autonomic reflex testing and neuropathy screening."),
}
_DEFAULT_RISK = ("REVIEW", "Result outside the standard classes; clinician review advised.")


def load_model():
    if not os.path.exists(ARTIFACT):
        raise SystemExit(
            "No trained model found. Run:  python -m neuroscreen.realtrain "
            "[--data data/paper2.csv]  first.")
    return joblib.load(ARTIFACT)


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


_ID_COLS = {"id", "patientid", "subject", "subjectid"}
_NAME_COLS = {"name", "patientname", "subjectname"}
_AGE_COLS = {"age"}
_SEX_COLS = {"sex", "gender"}


def _find_col(columns, wanted):
    for c in columns:
        if _norm(c) in wanted:
            return c
    return None


def extract_patient_meta(df: pd.DataFrame):
    """Pull identity columns (patient_id/name/age/sex) out of an input CSV, if
    present. These are stored alongside a screening but never treated as
    model features."""
    id_col = _find_col(df.columns, _ID_COLS)
    name_col = _find_col(df.columns, _NAME_COLS)
    age_col = _find_col(df.columns, _AGE_COLS)
    sex_col = _find_col(df.columns, _SEX_COLS)
    meta = []
    for i in range(len(df)):
        pid = str(df[id_col].iloc[i]) if id_col else f"row{i+1}"
        name = str(df[name_col].iloc[i]) if name_col and pd.notna(df[name_col].iloc[i]) else None
        age = None
        if age_col and pd.notna(df[age_col].iloc[i]):
            try:
                age = float(df[age_col].iloc[i])
            except (TypeError, ValueError):
                age = None
        sex = str(df[sex_col].iloc[i]) if sex_col and pd.notna(df[sex_col].iloc[i]) else None
        meta.append({"patient_id": pid, "name": name, "age": age, "sex": sex})
    return meta, {"id_col": id_col, "name_col": name_col, "age_col": age_col, "sex_col": sex_col}


def build_matrix(df: pd.DataFrame, model):
    """Align an input dataframe to the model's feature order.

    Matches each model feature to an input column by (1) normalised exact name
    -- which covers models whose features are the dataset's own column names --
    then (2) the canonical wearable synonym table. Missing markers are imputed
    with the training-cohort median and reported.
    """
    names = model["feature_names"]
    med = np.asarray(model["train_median"], float)
    input_by_norm = {}
    for c in df.columns:
        input_by_norm.setdefault(_norm(c), c)
    canon = ds.map_columns(df.columns)           # source -> canonical
    canon_inv = {v: k for k, v in canon.items()}  # canonical -> source

    X = np.empty((len(df), len(names)), float)
    used, imputed = [], []
    for j, key in enumerate(names):
        col = input_by_norm.get(_norm(key)) or canon_inv.get(key)
        if col is not None:
            v = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
            X[:, j] = np.where(np.isnan(v), med[j], v)
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
        "risk_level": RISK.get(pred, _DEFAULT_RISK)[0],
        "recommendation": RISK.get(pred, _DEFAULT_RISK)[1],
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


def write_template(path: str):
    """Write a blank input CSV: identity columns + whatever features the
    currently-trained model actually expects (falls back to the generic
    wearable schema if no model has been trained yet)."""
    id_cols = ["patient_id", "name", "age", "sex"]
    if os.path.exists(ARTIFACT):
        names = joblib.load(ARTIFACT)["feature_names"]
    else:
        names = ds.CANONICAL
    cols = id_cols + list(names)
    row = {**{k: "" for k in id_cols}, **{k: "" for k in names}}
    pd.DataFrame([row], columns=cols).to_csv(path, index=False)


def _print_patients(rows):
    if not rows:
        print("No patients on file yet. Screen someone with --input to add one.")
        return
    print(f"\n{len(rows)} patient(s) on file:")
    print(f"  {'PATIENT ID':<16}{'NAME':<20}{'AGE':<6}{'SEX':<6}{'SCREENINGS':<12}{'LAST RESULT':<14}LAST SCREENED")
    for r in rows:
        print(f"  {r['patient_id']:<16}{(r['name'] or '-'):<20}"
              f"{(str(r['age']) if r['age'] is not None else '-'):<6}"
              f"{(r['sex'] or '-'):<6}{r['n_screenings']:<12}"
              f"{(r['last_prediction'] or '-'):<14}{r['last_screened'] or '-'}")


def _print_history(patient_id, rows):
    if not rows:
        print(f"No screenings on file for patient '{patient_id}'.")
        return
    print(f"\nHistory for patient '{patient_id}' ({len(rows)} screening(s)), most recent first:")
    for r in rows:
        probs = json.loads(r["probabilities"])
        top = ", ".join(f"{k} {v*100:.0f}%" for k, v in probs.items())
        print(f"  {r['timestamp']}  ->  {r['prediction']:<10} "
              f"({r['confidence']*100:.0f}% conf, {r['risk_level']} risk)   [{top}]")


def main():
    ap = argparse.ArgumentParser(description="Screen a real person for neuropathy.")
    ap.add_argument("--input", help="CSV of subject wearable features")
    ap.add_argument("--template", metavar="PATH",
                    help="write a blank input template CSV and exit")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    ap.add_argument("--db", default=ndb.DEFAULT_PATH,
                    help=f"path to the SQLite results database (default: {ndb.DEFAULT_PATH})")
    ap.add_argument("--no-save", action="store_true", help="don't persist results to the database")
    ap.add_argument("--list-patients", action="store_true", help="list every patient on file, then exit")
    ap.add_argument("--history", metavar="PATIENT_ID",
                    help="print a patient's past screenings, then exit")
    args = ap.parse_args()

    if args.template:
        write_template(args.template)
        print(f"Wrote input template -> {args.template}")
        print("Fill one row per subject (patient_id/name/age/sex are optional but "
              "recommended for tracking), then run:")
        print(f"  python -m neuroscreen.detect --input {args.template}")
        return

    if args.list_patients:
        conn = ndb.connect(args.db)
        _print_patients(ndb.list_patients(conn))
        return

    if args.history:
        conn = ndb.connect(args.db)
        _print_history(args.history, ndb.patient_history(conn, args.history))
        return

    if not args.input:
        ap.error("provide --input <csv>, --template <csv>, --list-patients, or --history <id>")

    model = load_model()
    df = pd.read_csv(args.input)
    meta, meta_cols = extract_patient_meta(df)
    X, used, imputed = build_matrix(df, model)

    conn = None if args.no_save else ndb.connect(args.db)

    results = []
    for i in range(len(df)):
        res = screen_one(X[i], model)
        sid = meta[i]["patient_id"]
        results.append({"id": sid, **res})
        if conn is not None:
            ndb.upsert_patient(conn, sid, meta[i]["name"], meta[i]["age"], meta[i]["sex"])
            ndb.save_screening(conn, sid, model["source"], res, used, imputed)
        if not args.json:
            _print_report(sid, res, imputed)
            if conn is not None:
                n = len(ndb.patient_history(conn, sid))
                print(f"  Saved to {args.db}  ({n} screening(s) on file for {sid})")

    if args.json:
        print(json.dumps({"model_source": model["source"],
                          "features_used": used, "features_imputed": imputed,
                          "saved_to_db": (args.db if conn is not None else None),
                          "results": results}, indent=2))
    elif model["source"] == "synthetic":
        print("\n[!] Model trained on SYNTHETIC data -- results are a demonstration, "
              "not a clinical assessment. Retrain with the IEEE DataPort CSV via "
              "`python -m neuroscreen.realtrain --data <csv>` for real screening.")


if __name__ == "__main__":
    main()
