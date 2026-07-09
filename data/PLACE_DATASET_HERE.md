# Real datasets for training the screening model

## Primary (open, no login) — used by default

**PhysioNet — Cerebral Vasoregulation in Diabetes** (CC-BY 4.0)
28 diabetic + 22 control adults; non-invasive cardiac-autonomic tests + gait
speed. Task: **Diabetic vs Control**.

```
python -m neuroscreen.fetch_data      # downloads the ~130 KB summary CSV
python -m neuroscreen.realtrain --dataset vasoreg \
    --data data/physionet_diabetes/GE-71_Data_Summary_Table.csv
```

The two CSVs are committed under `data/physionet_diabetes/` for convenience
(see `SOURCE.md` there for attribution), so you can train immediately.

## Optional — Paper 2's own dataset (gated)

**IEEE DataPort, DOI 10.21227/f4jr-k711** — gait + plantar pressure + HRV,
labelled Healthy / DPN / CAN. Requires an IEEE DataPort **subscription login**,
so it can't be auto-downloaded. If you have access: download
`Dataset_and_approval_letter.zip`, extract the feature CSV to `data/paper2.csv`,
then:

```
python -m neuroscreen.realtrain --data data/paper2.csv
```

The loader (`neuroscreen/dataset.py`) maps common column-name spellings onto its
canonical gait/pressure/HRV schema and reports what it recognised.
