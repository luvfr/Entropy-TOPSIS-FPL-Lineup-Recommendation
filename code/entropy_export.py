import pandas as pd
import numpy as np
import os


def entropy_weights(df, criteria, is_benefit):
    X = df[criteria].values
    m, n = X.shape
    Z = np.zeros((m, n))
    for j in range(n):
        min_val = X[:, j].min()
        max_val = X[:, j].max()
        if is_benefit[j]:
            Z[:, j] = (X[:, j] - min_val) / (max_val - min_val + 1e-9)
        else:
            Z[:, j] = (max_val - X[:, j]) / (max_val - min_val + 1e-9)

    P = Z / (Z.sum(axis=0) + 1e-9)
    k = 1 / np.log(m + 1e-9)
    entropy = -k * (P * np.log(P + 1e-9)).sum(axis=0)
    d = 1 - entropy
    weights = d / (d.sum() + 1e-9)
    return weights, entropy, d

def save_entropy_table(codes, labels, is_benefit, entropy, d, weights,
                       position, period_label, output_dir):
    """Simpan tabel bobot entropy ke CSV."""
    jenis = ["Benefit" if b else "Cost" for b in is_benefit]
    df_out = pd.DataFrame({
        "Kode":                      codes,
        "Kriteria":                  labels,
        "Jenis":                     jenis,
        "Nilai Entropy (ej)":        np.round(entropy, 6),
        "Derajat Diferensiasi (dj)": np.round(d, 6),
        "Bobot (wj)":                np.round(weights, 6),
    })
    total_row = pd.DataFrame([{
        "Kode":                      "",
        "Kriteria":                  "TOTAL",
        "Jenis":                     "",
        "Nilai Entropy (ej)":        "",
        "Derajat Diferensiasi (dj)": "",
        "Bobot (wj)":                round(float(weights.sum()), 6),
    }])
    df_out = pd.concat([df_out, total_row], ignore_index=True)
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{period_label}_{position}_entropy_weights.csv"
    filepath = os.path.join(output_dir, filename)
    df_out.to_csv(filepath, index=False)
    print(f"  Tabel entropy {position} disimpan: {filename}")
    return df_out

def topsis(df, criteria, weights, is_benefit):
    df = df.copy()
    X = df[criteria].values
    m, n = X.shape
    R = X / np.sqrt((X**2).sum(axis=0) + 1e-9)
    V = R * weights
    A_pos = np.zeros(n)
    A_neg = np.zeros(n)
    for j in range(n):
        if is_benefit[j]:
            A_pos[j] = V[:, j].max()
            A_neg[j] = V[:, j].min()
        else:
            A_pos[j] = V[:, j].min()
            A_neg[j] = V[:, j].max()
    D_pos = np.sqrt(((V - A_pos) ** 2).sum(axis=1))
    D_neg = np.sqrt(((V - A_neg) ** 2).sum(axis=1))
    scores = D_neg / (D_pos + D_neg + 1e-9)
    df['topsis_score'] = scores
    df['rank'] = df['topsis_score'].rank(ascending=False, method='min')
    return df

# Folder proyek = induk dari folder code/. Dapat diganti lewat FPL_BASE_DIR.
BASE_DIR = os.environ.get(
    "FPL_BASE_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
DATA_DIR           = os.path.join(BASE_DIR, "data")
OUTPUT_ENTROPY_DIR = os.path.join(BASE_DIR, "results", "entropy_weights")

# Mapping folder → label periode
PERIOD_CONFIG = [
    ("GW 13 (Data GW 1-12)",  "GW13"),
    ("GW 16 (Data GW 1-15)",  "GW16"),
    ("GW 19 (Data GW 1-18)",  "GW19"),
    ("GW 22 (Data GW 1-21)",  "GW22"),
    ("GW 26 (Data GW 1-25)",  "GW26"),
    ("GW 30 (Data GW 1-29)",  "GW30"),
    ("GW 33 (Data GW 1-32)",  "GW33"),
    ("GW 36 (Data GW 1-35)",  "GW36"),
]

PROCESSED_SUBPATH  = os.path.join("processed", "musim 2025-2026")
PROCESSED_FILENAME = "players_processed_20252026.csv"
RANKING_SUBPATH    = os.path.join("output", "musim 2025-2026")

gk_cols   = ["clean_sheets", "saves", "penalties_saved", "bonus"]
gk_types  = [True, True, True, True]
gk_codes  = ["C1", "C2", "C3", "C4"]
gk_labels = ["Clean Sheets", "Saves", "Penyelamatan Penalti", "Bonus"]
 
def_cols   = ["clean_sheets", "goals", "assists", "bonus"]
def_types  = [True, True, True, True]
def_codes  = ["C1", "C2", "C3", "C4"]
def_labels = ["Clean Sheets", "Gol", "Assists", "Bonus"]
 
mid_cols   = ["clean_sheets", "goals", "assists", "bonus"]
mid_types  = [True, True, True, True]
mid_codes  = ["C1", "C2", "C3", "C4"]
mid_labels = ["Clean Sheets", "Gol", "Assists", "Bonus"]
 
fwd_cols   = ["threat", "goals", "assists", "bonus"]
fwd_types  = [True, True, True, True]
fwd_codes  = ["C1", "C2", "C3", "C4"]
fwd_labels = ["Threat", "Gol", "Assists", "Bonus"]
 
CRITERIA_CONFIG = {
    "GK":  (gk_cols,  gk_types,  gk_codes,  gk_labels),
    "DEF": (def_cols, def_types, def_codes, def_labels),
    "MID": (mid_cols, mid_types, mid_codes, mid_labels),
    "FWD": (fwd_cols, fwd_types, fwd_codes, fwd_labels),
}

for folder_name, period_label in PERIOD_CONFIG:
    print(f"\n{'='*60}")
    print(f"Periode: {period_label} ({folder_name})")
    print(f"{'='*60}")

    processed_path = os.path.join(
        DATA_DIR, folder_name, PROCESSED_SUBPATH, PROCESSED_FILENAME
    )
    ranking_dir = os.path.join(DATA_DIR, folder_name, RANKING_SUBPATH)

    if not os.path.exists(processed_path):
        print(f"✗ File tidak ditemukan: {processed_path}")
        print("  → Lewati periode ini.")
        continue

    df = pd.read_csv(processed_path)
    print(f"✓ Data dimuat: {len(df)} pemain")

    df_gk  = df[df['position'] == 'GK']
    df_def = df[df['position'] == 'DEF']
    df_mid = df[df['position'] == 'MID']
    df_fwd = df[df['position'] == 'FWD']

    os.makedirs(ranking_dir, exist_ok=True)
    df_gk.to_csv(os.path.join(ranking_dir,  "GK_processed_20252026.csv"),  index=False)
    df_def.to_csv(os.path.join(ranking_dir, "DEF_processed_20252026.csv"), index=False)
    df_mid.to_csv(os.path.join(ranking_dir, "MID_processed_20252026.csv"), index=False)
    df_fwd.to_csv(os.path.join(ranking_dir, "FWD_processed_20252026.csv"), index=False)
    print("✓ File per posisi disimpan")

    pos_dfs = {"GK": df_gk, "DEF": df_def, "MID": df_mid, "FWD": df_fwd}

    for pos, (cols, types, codes, labels) in CRITERIA_CONFIG.items():
        df_pos = pos_dfs[pos]
        if df_pos.empty:
            print(f"  ⚠ Tidak ada pemain posisi {pos}")
            continue

        weights, entropy, d = entropy_weights(df_pos, cols, types)

        save_entropy_table(
            codes        = codes,
            labels       = labels,
            is_benefit   = types,
            entropy      = entropy,
            d            = d,
            weights      = weights,
            position     = pos,
            period_label = period_label,
            output_dir   = OUTPUT_ENTROPY_DIR,
        )

        df_rank = topsis(df_pos, cols, weights, types)

        rank_path = os.path.join(ranking_dir, f"{pos}_ranking_20252026.csv")
        df_rank.to_csv(rank_path, index=False)

        top_idx = int(np.argmax(weights))
        print(f"  ✓ {pos:3s} | bobot tertinggi: {labels[top_idx]:<25} "
              f"({weights[top_idx]:.4f}) | ranking disimpan")

print(f"\n{'='*60}")
print("Semua periode selesai diproses!")
print(f"Entropy weights → {OUTPUT_ENTROPY_DIR}")
print(f"{'='*60}")