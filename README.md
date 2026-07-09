# NeuroScreen — Multi-Domain, Multi-Modal Diabetic Neuropathy Screening

A computer-aided screening prototype that stages **diabetic peripheral
neuropathy** and detects **cardiac autonomic neuropathy** by fusing two
complementary, non-invasive signal sources, with an interactive clinical
dashboard and explainable predictions.

It is built as a **novel synthesis of two research papers**:

| Paper | Modality | Method reproduced here |
|-------|----------|------------------------|
| Mengarelli et&nbsp;al., *IEEE TNSRE* 2024 — *Screening from Standing Balance* | Center-of-pressure (COP) sway during quiet standing | Multi-domain COP features + **kNN majority-voting ensemble** across AP / ML / statokinesigram components |
| Talha et&nbsp;al., *IEEE Access* 2026 — *Multimodal Wearable Sensors* | Gait (IMU) + plantar pressure + HRV during a walk | **Gradient-boosting** classifier + SHAP-style feature attribution |

### The idea / what's new
Paper 1 screens *peripheral* severity from posture; Paper 2 screens *type* of
neuropathy from a walk. Neither sees the other's blind spot. NeuroScreen fuses
both feature spaces into a single model and shows, quantitatively, **why the
fusion is necessary**:

- Posture COP ensemble on the three severity stages (NN / AN / SN): **86.5%**
  accuracy — reproduces Paper 1's "over 86%".
- The *same* posture model asked to also flag cardiac autonomic neuropathy
  (CAN): drops to **70.9%** — balance data is largely blind to autonomic
  involvement.
- Fused multi-modal gradient boosting on all four classes: **98.3%** — matches
  Paper 2's reported 98.27%.

All figures are leave-one-subject-out cross-validated.

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
- **Validated model performance** — switch between the three pathways and view
  per-class precision / sensitivity / specificity and the confusion matrix.

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
  (Paper 1).
- `build_gradient_boosting` — gradient boosting over the fused vector (Paper 2),
  whose `feature_importances_` drive the dashboard's explanations.

---

## Project layout
```
neuroscreen/
  signals.py     synthetic COP + wearable cohort generator
  features.py    multi-domain COP feature extraction
  models.py      kNN ensemble, gradient boosting, metrics
  train.py       pipeline: generate -> extract -> LOSO eval -> export
web/
  index.html     self-contained clinical dashboard (no external libraries)
  model.js       exported classifier + metrics (browser-runnable)
  model.json     same, as JSON
data/
  features.csv   full feature table for the cohort
run.py           train (if needed) + serve the dashboard
```

---

## Screening a real person (Paper 2 wearable pathway)

The wearable pathway is the realistic route to screening an actual subject: it
needs only IMUs, a plantar-pressure insole and an ECG/HRV strap — no force
plate — and Paper 2's labelled dataset can train it on real patients.

### Step 1 — get real labelled data
Paper 2's dataset is on IEEE DataPort
([DOI 10.21227/f4jr-k711](https://ieee-dataport.org/documents/ecg-imus-and-foot-plantar-pressure-signals-gait-and-health-monitoring)).
It is **subscription-gated**, so it cannot be auto-downloaded — log in, download
`Dataset_and_approval_letter.zip`, extract the feature CSV, and place it at
`data/paper2.csv` (any column names; the loader maps them — see
`neuroscreen/dataset.py::CANONICAL`).

### Step 2 — train on real patients
```bash
python -m neuroscreen.realtrain --data data/paper2.csv
```
This writes `models/realmodel.joblib` (scaler + gradient-boosting classifier +
feature schema + cohort stats) and prints leave-one-subject-out accuracy and
per-class sensitivity/specificity. Without `--data` it trains on the synthetic
frame so you can try the flow immediately (results are clearly flagged as a
demonstration).

### Step 3 — screen a subject
Capture the subject's gait / plantar-pressure / HRV markers into a one-row CSV
(use the template), then run the detector:
```bash
python -m neuroscreen.detect --template data/subject.csv   # blank input to fill in
python -m neuroscreen.detect --input   data/subject.csv    # screen
python -m neuroscreen.detect --input   data/subject.csv --json
```
Output: predicted class (Healthy / DPN / CAN), confidence, clinical risk level,
the markers that most drove the decision (deviation from the healthy reference
weighted by model importance), and a referral recommendation. Missing markers
are imputed with the training-cohort median and reported.

### From sensors to a CSV
`detect.py` consumes *features*, not raw signals. A real capture rig computes
them on-device or in a short pre-processing step: gait spatiotemporal +
kinematic features from the two IMUs (heel-strike / toe-off via peak detection,
per Paper 2), peak/mean pressure per foot region from the insole, and
time/frequency HRV metrics (SDRR, rMSSD, SD1/SD2, LF/HF) from RR intervals via
Pan–Tompkins. Emit one row per subject with those columns and feed it in.

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
