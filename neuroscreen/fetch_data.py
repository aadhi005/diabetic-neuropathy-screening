"""
Download the open-access real dataset used for real-person screening.

PhysioNet *Cerebral Vasoregulation in Diabetes* (CC-BY 4.0, no login required).
Only the small per-subject summary table and data dictionary are needed for the
feature-level screening model; the 3 GB raw-signal archive is not downloaded.

    python -m neuroscreen.fetch_data
"""

from __future__ import annotations

import os
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "data", "physionet_diabetes")
BASE = ("https://physionet.org/files/cerebral-vasoreg-diabetes/1.0.0/"
        "Data_Description")
FILES = ["GE-71_Data_Summary_Table.csv", "GE-71_Data_Dictionary.csv"]


def main():
    os.makedirs(DEST, exist_ok=True)
    for name in FILES:
        out = os.path.join(DEST, name)
        if os.path.exists(out):
            print(f"  exists: {name}")
            continue
        url = f"{BASE}/{name}"
        print(f"  downloading {name} ...")
        urllib.request.urlretrieve(url, out)
    print(f"\nDone -> {os.path.relpath(DEST, ROOT)}")
    print("Train on it with:")
    print("  python -m neuroscreen.realtrain --dataset vasoreg "
          "--data data/physionet_diabetes/GE-71_Data_Summary_Table.csv")


if __name__ == "__main__":
    main()
