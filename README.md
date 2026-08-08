# NeuroScreen — Multi-Domain, Multi-Modal Diabetic Neuropathy Screening

A computer-aided screening prototype that stages **diabetic peripheral
neuropathy** and detects **cardiac autonomic neuropathy** by fusing two
complementary, non-invasive signal sources, with an interactive clinical
dashboard and explainable predictions.

It is built as a **novel synthesis of two research papers**:

| Paper | Modality | Method reproduced here |
|-------|----------|------------------------|
| **Paper 1** (verified) — A. Mengarelli, A. Tigrini, F. Verdini, M. Scattolini, R. Mobarak, L. Burattini, R. A. Rabini, S. Fioretti, "A Computer-Aided Screening Solution for the Identification of Diabetic Neuropathy From Standing Balance by Leveraging Multi-Domain Features," *IEEE Trans. Neural Syst. Rehabil. Eng.*, vol. 32, pp. 2388–2397, 2024 | Center-of-pressure (COP) sway during quiet standing | Multi-domain COP features + **kNN majority-voting ensemble** across AP / ML / statokinesigram components, and **both diagnosis pathways** (DP-1 and DP-2) |
| **Paper 2** (citation unverified — see note) — *Physiological Features for Classification of Different Types of Peripheral Neuropathy Using Multimodal Wearable Sensors* | Gait (IMU) + plantar pressure + HRV during a walk | **Gradient-boosting** classifier + SHAP-style feature attribution |

> **Citation note.** Paper 1's bibliographic details above were read directly
> from the PDF. Paper 2's source PDF is not present in this workspace, so its
> authors, venue, year and reported accuracy have **not** been verified and are
> deliberately omitted rather than guessed. Its linked dataset (IEEE DataPort,
> DOI `10.21227/f4jr-k711`) was independently confirmed to exist and to carry
> DPN/CAN labels. Re-add the full citation once the PDF is available.

### The idea / what's new
Paper 1 screens *peripheral* severity from posture; Paper 2 screens *type* of
neuropathy from a walk. Neither sees the other's blind spot. NeuroScreen fuses
both feature spaces into a single model and shows, quantitatively, **why the
fusion is necessary**:

- **DP-1** — Paper 1's single 3-class model over the severity stages
  (NN / AN / SN): **86.5%**, reproducing the paper's "over 86%".
- **DP-2** — Paper 1's two-stage cascade, `(NN+AN) vs SN` then `NN vs AN`:
  **95.2%**. The paper reports this pathway beating DP-1 (>97%), which is what
  makes *asymptomatic* detection viable; DP-1's ~86% does not support it.
- The *same* posture model asked to also flag cardiac autonomic neuropathy
  (CAN): drops to **70.9%** — balance data is largely blind to autonomic
  involvement.
- Fused multi-modal gradient boosting on all four classes: **98.3%**.

All figures are leave-one-subject-out cross-validated.

> **These four numbers come from synthetic data.** They demonstrate that the
> pipeline reproduces the papers' methods and reported *behaviour*; they are not
> evidence of real-world accuracy. On real patients the same pipeline scores
> **70.0%** (see *Screening a real person* below) — that gap is the honest
> headline, and the dashboard labels every figure with its provenance.

---

## Quick start

```bash
pip install -r requirements.txt
python run.py            # trains if needed, then opens the dashboard
```

Then open <http://localhost:8777>. The dashboard also opens directly by
double-clicking `web/index.html` (the model is bundled as `web/model.js`, so no
server is strictly required).

To regenerate the cohort, features and model:

```bash
python -m neuroscreen.train      # ~2 min; writes web/model.json + data/features.csv
```

Run the tests:

```bash
python -m pytest tests/ -q       # 42 tests, ~17 s
```

---

## What the dashboard does
- **Virtual patient** — load a representative case (NN / AN / SN / CAN) or drive
  four physiological dials (postural instability, gait impairment, plantar
  pressure, autonomic/HRV loss).
- **Screening result** — class probabilities from a distance-weighted kNN over
  the reference cohort, a confidence gauge, and a clinical risk banner.
- **Standing-balance sway** — a live statokinesigram (COP planar trajectory)
  with its 95% confidence ellipse.
- **Why this decision** — importance-weighted feature attribution showing which
  markers pushed the case toward health vs. neuropathy.
- **Patient profile** — the header avatar opens a record form (name, ID, age,
  sex, diabetes type, HbA1c, notes). Records persist in browser storage, can
  carry the current screening result, and export as a CSV that
  `neuroscreen.detect` reads straight into the SQLite database.
- **Validated model performance** — switch between the pathways (DP-1, DP-2,
  all-classes, fused) **and the real-patient model**, with per-class precision /
  sensitivity / specificity and the confusion matrix. A provenance banner under
  the tabs states whether the figures on screen are synthetic or from real
  patients, so the two can never be confused.

---

## How it works

### Signals (`neuroscreen/signals.py`)
The source datasets are not distributed locally (Paper 1's cohort is private;
Paper 2's is on IEEE DataPort), so a **physiologically-grounded synthetic
cohort** is generated whose *class-dependent behaviour* matches the papers:
COP sway modelled with a bounded Ornstein–Uhlenbeck process (open-/closed-loop
postural control), plus gait / plantar-pressure / HRV markers with severity- and
autonomy-dependent shifts. Deterministic given a seed.

### Features (`neuroscreen/features.py`)
Faithful implementations of Paper 1's three feature families, computed for the
APc, MLc and statokinesigram (STKc) components:
1. **Universal balance descriptors** — SL (low-frequency spectral slope), PS50
   (median frequency), ANG & FLAT (95%-ellipse geometry).
2. **Stabilogram Diffusion Function** (Collins & De Luca) — CRT, MSD, DS, DL,
   HS, HL, K, TL.
3. **Structural / complexity** — Recurrence Quantification Analysis (RR, DET,
   RT, AVDL, MLL, ENT, TND, LAM, MVL, TT) and sample entropy (SAEN).

These 65 COP features are fused with 13 wearable gait/pressure/HRV markers →
**78 features per subject**.

### Models (`neuroscreen/models.py`)
- `ComponentEnsemble` — a kNN per COP component combined by majority / soft vote
  (Paper 1), optionally with per-component backward feature selection.
- `DiagnosisPathway2` — Paper 1's DP-2 cascade: stage 1 separates symptomatic
  neuropathy `(NN+AN) vs SN`, stage 2 then splits `NN vs AN` among the rest.
  Stage 2 is fitted only on non-symptomatic patients, matching deployment.
- `backward_feature_selection` — the paper's B-FS: greedy backward elimination
  repeated over random half-splits, keeping features chosen by at least half the
  runs. Off by default for headline figures: at ~45 s per COP block it cannot be
  nested inside leave-one-subject-out (~8 h), and running it once over the whole
  cohort would leak test subjects into selection and inflate the result.
- `build_gradient_boosting` — gradient boosting over the fused vector (Paper 2),
  whose `feature_importances_` drive the dashboard's explanations.

---

## Project layout
```
neuroscreen/
  signals.py      synthetic COP + wearable cohort generator
  features.py     multi-domain COP feature extraction
  models.py       kNN ensemble, DP-2 cascade, B-FS, gradient boosting, metrics
  train.py        pipeline: generate -> extract -> LOSO eval -> export
  dataset.py      tolerant wearable-CSV ingestion (synonyms + units)
  realdata.py     loaders for specific public datasets
  realtrain.py    train + persist the real-person model
  export_real.py  real model -> web/realmodel.js for the dashboard
  detect.py       screen a subject; writes results to the database
  db.py           SQLite storage for patients and screenings
  fetch_data.py   download the open PhysioNet dataset
web/
  index.html      self-contained clinical dashboard (no external libraries)
  model.js        exported synthetic classifier + metrics (browser-runnable)
  realmodel.js    exported real-patient model + metrics
tests/            42 tests: features, ingestion, DP-2, metrics, database
data/
  features.csv    full feature table for the cohort
run.py            train (if needed) + serve the dashboard
```

---

## Screening a real person (real open data)

Paper 2's own dataset (IEEE DataPort, DOI 10.21227/f4jr-k711) is
**subscription-gated** and could not be used. Instead the real-person pathway is
trained on a fully open dataset with the same clinical intent:

> **PhysioNet — Cerebral Vasoregulation in Diabetes** (CC-BY 4.0, no login):
> 28 type-2 diabetic and 22 control adults with non-invasive **cardiac-autonomic
> function** tests (Valsalva ratio, head-up-tilt HR/BP responses, orthostatic BP
> change) and gait speed.

This makes the real task **Diabetic (autonomic-affected) vs Control** — the
autonomic axis of diabetic neuropathy, screenable with a bedside/wearable
autonomic-reflex protocol.

### Step 1 — get the data (open, ~130 KB)
```bash
python -m neuroscreen.fetch_data
```
Downloads the per-subject summary CSV from PhysioNet (see
`data/physionet_diabetes/SOURCE.md` for attribution).

### Step 2 — train on real patients
```bash
python -m neuroscreen.realtrain --dataset vasoreg \
    --data data/physionet_diabetes/GE-71_Data_Summary_Table.csv
```
Writes `models/realmodel.joblib` and prints leave-one-subject-out metrics.
**Honest performance: ~70% LOSO accuracy** (majority-class baseline 56%) — real
biomedical data at n=50 is hard, unlike the synthetic demo. On small cohorts the
trainer auto-selects a sparse elastic-net logistic model; larger sets use
gradient boosting. Running `realtrain` with no `--data` falls back to a
synthetic frame, clearly flagged.

### Step 3 — screen a subject
```bash
python -m neuroscreen.detect --template data/subject.csv   # blank input, model-matched columns
python -m neuroscreen.detect --input data/subject.csv      # report
python -m neuroscreen.detect --input data/subject.csv --json
```
Output per subject: predicted class, confidence, clinical risk level, the
markers that most drove the decision (deviation from the control reference
weighted by model importance), and a recommendation. The detector matches your
CSV's columns to the model's features by name — tolerating synonyms and units,
so `RMSSD`, `rmssd (ms)` and `SDNN [ms]` all resolve — and imputes anything
missing with the training median, reporting what it imputed.

### Step 4 — show the real model in the dashboard
```bash
python -m neuroscreen.export_real   # writes web/realmodel.js
```
Adds a **Real patients** tab to the dashboard's metrics panel next to the
synthetic pathways, so the 70% real result and the 98.3% synthetic one are
visible side by side and each is labelled with its data source.

### Patient database
Every screening is written to a local **SQLite** database
(`data/neuroscreen.db`, override with `--db`), so results accumulate per
patient instead of being a one-off printout. The input CSV may carry optional
identity columns — `patient_id`, `name`, `age`, `sex` — which are stored with
the result but never used as model features.

```bash
python -m neuroscreen.detect --input data/subject.csv --no-save  # screen without storing
python -m neuroscreen.detect --list-patients                     # everyone on file
python -m neuroscreen.detect --history P001                      # one patient over time
```

Two tables (`neuroscreen/db.py`): `patients` holds identity, `screenings` holds
one row per run (timestamp, model source, prediction, confidence, risk,
probabilities, top factors, which features were measured vs imputed).

> The `.db` file is **git-ignored** — it can hold real names and health data and
> must never be committed to a public repository.

### Swapping in a different dataset
`neuroscreen/realdata.py` holds the loader + curated feature list; add a loader
to `LOADERS` and pass `--dataset <key>`. Paper-2-style wearable CSVs
(gait/pressure/HRV, Healthy/DPN/CAN) are handled by the tolerant
`neuroscreen/dataset.py` schema if you obtain that data.

---

## Extending the balance pathway
For the COP side, swap `build_cohort()` for a loader of real force-plate
recordings, keep `extract_cop_features()` unchanged, and re-run
`python -m neuroscreen.train`. The dashboard consumes whatever `model.js` the
pipeline exports.

> **Disclaimer.** Research / educational prototype. The bundled model is trained
> on synthetic data and is **not a medical device** — it flags people who should
> receive a proper nerve-conduction / autonomic workup; it does not diagnose.
> Real clinical use requires training on real labelled data, independent-cohort
> validation, ethics approval, and regulatory clearance.
