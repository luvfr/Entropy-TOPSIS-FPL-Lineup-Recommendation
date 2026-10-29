import pandas as pd
import numpy as np
import os

# Folder proyek = induk dari folder code/. Dapat diganti lewat FPL_BASE_DIR.
BASE_DIR = os.environ.get(
    "FPL_BASE_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
DATA_DIR = os.path.join(BASE_DIR, "data")
RANKING_SUBPATH = os.path.join("output", "musim 2025-2026")

PERIODS = [
    "GW 13 (Data GW 1-12)", "GW 16 (Data GW 1-15)",
    "GW 19 (Data GW 1-18)", "GW 22 (Data GW 1-21)",
    "GW 26 (Data GW 1-25)", "GW 30 (Data GW 1-29)",
    "GW 33 (Data GW 1-32)", "GW 36 (Data GW 1-35)",
]

# Kriteria per posisi (harga tidak dipakai sebagai kriteria), sama dengan entropy_export.py
CRITERIA_CONFIG = {
    "GK":  (["clean_sheets", "saves", "penalties_saved", "bonus"], [True, True, True, True]),
    "DEF": (["clean_sheets", "goals", "assists", "bonus"], [True, True, True, True]),
    "MID": (["clean_sheets", "goals", "assists", "bonus"], [True, True, True, True]),
    "FWD": (["threat", "goals", "assists", "bonus"], [True, True, True, True]),
}

RANKING_FILES = {
    "GK": "GK_ranking_20252026.csv", "DEF": "DEF_ranking_20252026.csv",
    "MID": "MID_ranking_20252026.csv", "FWD": "FWD_ranking_20252026.csv",
}

def topsis_equal_weight(df, criteria, is_benefit):
    X = df[criteria].values
    m, n = X.shape
    weights = np.full(n, 1.0 / n)  # bobot sama rata, bukan dari Entropy
    R = X / np.sqrt((X**2).sum(axis=0) + 1e-9)
    V = R * weights
    A_pos, A_neg = np.zeros(n), np.zeros(n)
    for j in range(n):
        if is_benefit[j]:
            A_pos[j], A_neg[j] = V[:, j].max(), V[:, j].min()
        else:
            A_pos[j], A_neg[j] = V[:, j].min(), V[:, j].max()
    D_pos = np.sqrt(((V - A_pos) ** 2).sum(axis=1))
    D_neg = np.sqrt(((V - A_neg) ** 2).sum(axis=1))
    return D_neg / (D_pos + D_neg + 1e-9)

def main():
    for folder_name in PERIODS:
        ranking_dir = os.path.join(DATA_DIR, folder_name, RANKING_SUBPATH)
        print(f"\n{folder_name}")
        for pos, filename in RANKING_FILES.items():
            path = os.path.join(ranking_dir, filename)
            if not os.path.exists(path):
                print(f"  x {pos}: file tidak ditemukan")
                continue
            df = pd.read_csv(path)

            df['baseline1_score'] = df['total_points']

            criteria, is_benefit = CRITERIA_CONFIG[pos]
            df['baseline2_score'] = topsis_equal_weight(df, criteria, is_benefit)

            df.to_csv(path, index=False)
            print(f"  v {pos}: baseline1_score & baseline2_score ditambahkan")

    print("\nSelesai! Selanjutnya jalankan seleksi_tim_baseline.py")

if __name__ == "__main__":
    main()