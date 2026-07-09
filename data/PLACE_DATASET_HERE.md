# Where to put the real dataset

The real-person screening model trains on **Paper 2's wearable feature
dataset** (gait + plantar pressure + HRV, labelled Healthy / DPN / CAN):

- IEEE DataPort — DOI **10.21227/f4jr-k711**
- <https://ieee-dataport.org/documents/ecg-imus-and-foot-plantar-pressure-signals-gait-and-health-monitoring>

This dataset is **behind an IEEE DataPort subscription login**, so it cannot be
downloaded automatically. Steps:

1. Log in to IEEE DataPort and download `Dataset_and_approval_letter.zip`.
2. Extract it and find the feature CSV inside.
3. Save that CSV here as **`data/paper2.csv`**.
4. Train on real patients:
   ```
   python -m neuroscreen.realtrain --data data/paper2.csv
   ```

The loader (`neuroscreen/dataset.py`) is tolerant of column naming — it maps
common spellings of gait / pressure / HRV markers onto its canonical schema and
reports exactly what it recognised. If some columns aren't picked up, add their
names to `SYNONYMS` in that file.

Until then, `python -m neuroscreen.realtrain` (no `--data`) trains on a
synthetic stand-in so the pipeline is fully runnable, with results clearly
flagged as a demonstration.
