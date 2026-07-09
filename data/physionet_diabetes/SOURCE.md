# Data source & attribution

The two CSVs in this folder are the per-subject **summary table** and **data
dictionary** from:

> **Cerebral Vasoregulation in Diabetes** (version 1.0.0). PhysioNet.
> Available: https://physionet.org/content/cerebral-vasoreg-diabetes/1.0.0/

- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0) —
  redistribution permitted with attribution.
- **Cohort:** 86 adults aged 55–75; the summary table covers 50 with complete
  entries (28 type-2 diabetic, 22 control).
- **What we use:** non-invasive **cardiac-autonomic function** measures
  (Valsalva ratio, head-up-tilt HR/BP responses, orthostatic BP change) and
  gait speed — see `neuroscreen/realdata.py::VASOREG_FEATURES`.
- **What we do NOT use:** labs, MRI perfusion and glucose fields (they would
  make diabetic-vs-control trivial / are not wearable-obtainable). The raw
  3 GB signal archive is not needed.

Re-download anytime with `python -m neuroscreen.fetch_data`.

PhysioNet citation: Goldberger AL, et al. *PhysioBank, PhysioToolkit, and
PhysioNet.* Circulation. 2000;101(23):e215–e220.
