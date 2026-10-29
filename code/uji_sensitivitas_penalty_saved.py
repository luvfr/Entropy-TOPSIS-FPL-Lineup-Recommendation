import pandas as pd
import numpy as np
import os
from scipy.stats import spearmanr

# Folder proyek = induk dari folder code/. Dapat diganti lewat FPL_BASE_DIR.
BASE_DIR = os.environ.get(
    "FPL_BASE_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_SUBPATH  = os.path.join("processed", "musim 2025-2026")
PROCESSED_FILENAME = "players_processed_20252026.csv"

PERIODS = [
    "GW 13 (Data GW 1-12)", "GW 16 (Data GW 1-15)",
    "GW 19 (Data GW 1-18)", "GW 22 (Data GW 1-21)",
    "GW 26 (Data GW 1-25)", "GW 30 (Data GW 1-29)",
    "GW 33 (Data GW 1-32)", "GW 36 (Data GW 1-35)",
]

# Kriteria GK (harga tidak dipakai sebagai kriteria)
GK_COLS_FULL = ["clean_sheets", "saves", "penalties_saved", "bonus"]
GK_TYPES_FULL = [True, True, True, True]

# Versi tanpa penalty saved
GK_COLS_NOPEN = ["clean_sheets", "saves", "bonus"]
GK_TYPES_NOPEN = [True, True, True]

def entropy_weights(df, criteria, is_benefit):
    X = df[criteria].values
    m, n = X.shape
    Z = np.zeros((m, n))
    for j in range(n):
        min_val, max_val = X[:, j].min(), X[:, j].max()
        if is_benefit[j]:
            Z[:, j] = (X[:, j] - min_val) / (max_val - min_val + 1e-9)
        else:
            Z[:, j] = (max_val - X[:, j]) / (max_val - min_val + 1e-9)
    P = Z / (Z.sum(axis=0) + 1e-9)
    k = 1 / np.log(m + 1e-9)
    entropy = -k * (P * np.log(P + 1e-9)).sum(axis=0)
    d = 1 - entropy
    weights = d / (d.sum() + 1e-9)
    return weights

def topsis(df, criteria, weights, is_benefit):
    df = df.copy()
    X = df[criteria].values
    m, n = X.shape
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
    scores = D_neg / (D_pos + D_neg + 1e-9)
    df['topsis_score'] = scores
    df['rank'] = df['topsis_score'].rank(ascending=False, method='min')
    return df

def main():
    all_period_results = []

    for folder_name in PERIODS:
        path = os.path.join(DATA_DIR, folder_name, PROCESSED_SUBPATH, PROCESSED_FILENAME)
        if not os.path.exists(path):
            print(f"✗ {folder_name}: file tidak ditemukan, dilewati")
            continue

        df = pd.read_csv(path)
        df_gk = df[df['position'] == 'GK'].reset_index(drop=True)

        w_full = entropy_weights(df_gk, GK_COLS_FULL, GK_TYPES_FULL)
        df_full = topsis(df_gk, GK_COLS_FULL, w_full, GK_TYPES_FULL)

        w_nopen = entropy_weights(df_gk, GK_COLS_NOPEN, GK_TYPES_NOPEN)
        df_nopen = topsis(df_gk, GK_COLS_NOPEN, w_nopen, GK_TYPES_NOPEN)

        comp = pd.DataFrame({
            'nama': df_gk['first_name'] + ' ' + df_gk['second_name'],
            'rank_dengan_penalti': df_full['rank'].values,
            'rank_tanpa_penalti': df_nopen['rank'].values,
        })
        comp['selisih_rank'] = (comp['rank_dengan_penalti'] - comp['rank_tanpa_penalti']).abs()

        # Korelasi Spearman antar 2 ranking (1.0 = identik, 0 = tidak berkorelasi)
        rho, _ = spearmanr(comp['rank_dengan_penalti'], comp['rank_tanpa_penalti'])

        top5_full = set(comp.nsmallest(5, 'rank_dengan_penalti')['nama'])
        top5_nopen = set(comp.nsmallest(5, 'rank_tanpa_penalti')['nama'])
        top5_overlap = len(top5_full & top5_nopen)

        max_shift = comp['selisih_rank'].max()
        max_shift_player = comp.loc[comp['selisih_rank'].idxmax(), 'nama']

        print(f"{folder_name}")
        print(f"  Korelasi Spearman (ranking): {rho:.4f}")
        print(f"  Overlap Top-5: {top5_overlap}/5 pemain sama")
        print(f"  Pergeseran rank terbesar: {max_shift_player} (selisih {int(max_shift)} peringkat)")
        print()

        all_period_results.append({
            'periode': folder_name, 'spearman_rho': rho,
            'top5_overlap': top5_overlap, 'max_shift': max_shift,
            'max_shift_player': max_shift_player,
        })

    df_summary = pd.DataFrame(all_period_results)
    df_summary.to_csv("sensitivitas_penalty_saved_summary.csv", index=False)
    print("="*60)
    print("Ringkasan semua periode:")
    print(df_summary.to_string(index=False))
    print(f"\nRata-rata korelasi Spearman: {df_summary['spearman_rho'].mean():.4f}")
    print("Disimpan ke sensitivitas_penalty_saved_summary.csv")

if __name__ == "__main__":
    main()