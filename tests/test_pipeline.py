"""
Tests for the ingestion / persistence layers.

The detector has to accept CSVs it has never seen, written by other people
with other column names, and must never silently invent data -- these cover
the paths where that could go wrong.
"""

import numpy as np
import pandas as pd
import pytest

from neuroscreen import dataset as ds
from neuroscreen import db as ndb
from neuroscreen.models import (ComponentEnsemble, DiagnosisPathway2,
                                confusion, per_class_metrics)


# --------------------------------------------------------------------------
# Column mapping
# --------------------------------------------------------------------------
@pytest.mark.parametrize("given,expected", [
    ("RMSSD", "hrv_rmssd"),
    ("rmssd (ms)", "hrv_rmssd"),          # units in the header
    ("SDNN", "hrv_sdrr"),
    ("SDNN [ms]", "hrv_sdrr"),            # square-bracket units
    ("Cadence", "gait_cadence"),
    ("forefoot_max", "press_forefoot_peak"),
    ("Forefoot peak (kPa)", "press_forefoot_peak"),
    ("LF/HF ratio", "hrv_lfhf"),
])
def test_synonyms_map_to_canonical(given, expected):
    """Third-party CSVs spell these a dozen ways; all must collapse to one key."""
    assert ds.map_columns([given]) == {given: expected}


def test_unknown_columns_are_ignored_not_guessed():
    assert ds.map_columns(["patient_weight", "room_temperature"]) == {}


def test_label_column_found_by_name_and_by_values():
    by_name = pd.DataFrame({"diagnosis": ["DPN"], "rmssd": [30.0]})
    assert ds.find_label_column(by_name) == "diagnosis"
    by_values = pd.DataFrame({"col_x": ["Healthy", "DPN"], "rmssd": [30.0, 20.0]})
    assert ds.find_label_column(by_values) == "col_x"


@pytest.mark.parametrize("raw,canon", [
    ("healthy", "Healthy"), ("Control", "Healthy"), ("normal", "Healthy"),
    ("DPN", "DPN"), ("diabetic", "DPN"),
    ("CAN", "CAN"), ("autonomic", "CAN"),
])
def test_label_normalisation(raw, canon):
    assert ds.normalise_labels(pd.Series([raw])).iloc[0] == canon


def test_load_csv_reports_missing_features(tmp_path):
    """A CSV with only some markers must train on those and say what is absent."""
    p = tmp_path / "partial.csv"
    pd.DataFrame({"rmssd": [30.0, 20.0, 35.0, 18.0],
                  "cadence": [110.0, 95.0, 112.0, 92.0],
                  "label": ["Healthy", "DPN", "Healthy", "DPN"]}).to_csv(p, index=False)
    X, y, names, report = ds.load_csv(str(p), verbose=False)
    assert X.shape == (4, 2)
    assert set(names) == {"hrv_rmssd", "gait_cadence"}
    assert "hrv_sdrr" in report["features_missing"]
    assert report["class_counts"]["Healthy"] == 2


def test_load_csv_raises_without_label_column(tmp_path):
    p = tmp_path / "nolabel.csv"
    pd.DataFrame({"rmssd": [30.0], "cadence": [110.0]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="label"):
        ds.load_csv(str(p), verbose=False)


def test_template_round_trips_through_the_loader(tmp_path):
    """The template we hand clinicians must be readable by our own loader."""
    p = tmp_path / "template.csv"
    ds.write_template(str(p))
    cols = pd.read_csv(p).columns
    assert "label" in cols
    assert set(ds.CANONICAL).issubset(set(cols))


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def test_per_class_metrics_on_a_known_confusion():
    y_true = np.array(["A", "A", "A", "B", "B"])
    y_pred = np.array(["A", "A", "B", "B", "B"])
    acc, rows = per_class_metrics(y_true, y_pred, ["A", "B"])
    assert acc == pytest.approx(4 / 5)
    assert rows["A"]["sensitivity"] == pytest.approx(2 / 3)
    assert rows["A"]["precision"] == pytest.approx(1.0)
    assert rows["B"]["sensitivity"] == pytest.approx(1.0)
    assert rows["B"]["specificity"] == pytest.approx(2 / 3)


def test_confusion_rows_are_truth():
    y_true = np.array(["A", "A", "B"])
    y_pred = np.array(["A", "B", "B"])
    np.testing.assert_array_equal(confusion(y_true, y_pred, ["A", "B"]),
                                  np.array([[1, 1], [0, 1]]))


# --------------------------------------------------------------------------
# DP-2 cascade behaviour
# --------------------------------------------------------------------------
def _toy_cop(n=24, seed=0):
    """Separable 3-class toy set with the APc/MLc/STKc naming the models expect."""
    rng = np.random.default_rng(seed)
    names, blocks = [], []
    for comp in ("APc", "MLc", "STKc"):
        names += [f"{comp}_f{i}" for i in range(3)]
    y, rows = [], []
    for k, lab in enumerate(["NN", "AN", "SN"]):
        for _ in range(n // 3):
            rows.append(rng.normal(loc=k * 4.0, scale=0.3, size=len(names)))
            y.append(lab)
    return np.array(rows), np.array(y), names


def test_dp2_predicts_only_valid_labels():
    X, y, names = _toy_cop()
    pred = DiagnosisPathway2(names, k=3).fit(X, y).predict(X)
    assert set(pred).issubset({"NN", "AN", "SN"})
    assert len(pred) == len(y)


def test_dp2_separates_a_clearly_separable_cohort():
    X, y, names = _toy_cop()
    pred = DiagnosisPathway2(names, k=3).fit(X, y).predict(X)
    assert np.mean(pred == y) > 0.9


def test_dp2_stage2_never_sees_symptomatic_patients():
    """Stage 2 decides NN vs AN only; SN must be filtered out before it."""
    X, y, names = _toy_cop()
    m = DiagnosisPathway2(names, k=3).fit(X, y)
    assert set(m.stage1.classes_) == {"SN", DiagnosisPathway2.NEG}
    assert set(m.stage2.classes_) == {"NN", "AN"}


def test_component_ensemble_uses_only_its_own_block():
    X, y, names = _toy_cop()
    ens = ComponentEnsemble(names, k=3).fit(X, y)
    for comp, idx in ens.blocks.items():
        assert all(names[i].startswith(comp + "_") for i in idx)


# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------
def test_patient_and_screening_round_trip(tmp_path):
    conn = ndb.connect(str(tmp_path / "t.db"))
    ndb.upsert_patient(conn, "P1", name="Ravi", age=62, sex="Male")
    result = {"prediction": "DPN", "confidence": 0.83, "risk_level": "HIGH",
              "recommendation": "Refer", "probabilities": {"DPN": 0.83},
              "top_factors": [{"feature": "hrv_rmssd", "z": -2.1}]}
    ndb.save_screening(conn, "P1", "test-model", result, ["hrv_rmssd"], [])

    hist = ndb.patient_history(conn, "P1")
    assert len(hist) == 1 and hist[0]["prediction"] == "DPN"
    assert hist[0]["confidence"] == pytest.approx(0.83)

    listed = ndb.list_patients(conn)
    assert listed[0]["patient_id"] == "P1"
    assert listed[0]["n_screenings"] == 1
    assert listed[0]["last_prediction"] == "DPN"


def test_upsert_does_not_blank_existing_fields(tmp_path):
    """A follow-up visit that only supplies an ID must not erase the name."""
    conn = ndb.connect(str(tmp_path / "t.db"))
    ndb.upsert_patient(conn, "P1", name="Ravi", age=62, sex="Male")
    ndb.upsert_patient(conn, "P1")
    row = conn.execute("SELECT name, age FROM patients WHERE patient_id='P1'").fetchone()
    assert row["name"] == "Ravi" and row["age"] == 62


def test_history_accumulates_across_visits(tmp_path):
    conn = ndb.connect(str(tmp_path / "t.db"))
    ndb.upsert_patient(conn, "P1", name="Ravi")
    for pred in ("Healthy", "DPN"):
        ndb.save_screening(conn, "P1", "m", {
            "prediction": pred, "confidence": 0.7, "risk_level": "LOW",
            "recommendation": "-", "probabilities": {}, "top_factors": []}, [], [])
    assert len(ndb.patient_history(conn, "P1")) == 2
    assert ndb.list_patients(conn)[0]["n_screenings"] == 2
