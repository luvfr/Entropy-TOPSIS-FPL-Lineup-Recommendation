import pandas as pd
import numpy as np
import os
import unicodedata
import re
from rapidfuzz import process, fuzz


# Folder proyek = induk dari folder code/. Dapat diganti lewat FPL_BASE_DIR.
BASE_DIR = os.environ.get(
    "FPL_BASE_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
DATA_DIR    = os.path.join(BASE_DIR, "data")
VAASTAV_DIR = os.path.join(BASE_DIR, "results", "vaastav")

WINDOW_SIZE = 3  # jumlah GW yang dirata-rata

# Periode -> GW batas (data boleh dipakai sampai GW ini)
# Window = [GW_batas - WINDOW_SIZE + 1, GW_batas]
PERIOD_CUTOFF = {
    "GW 13 (Data GW 1-12)": 12,
    "GW 16 (Data GW 1-15)": 15,
    "GW 19 (Data GW 1-18)": 18,
    "GW 22 (Data GW 1-21)": 21,
    "GW 26 (Data GW 1-25)": 25,
    "GW 30 (Data GW 1-29)": 29,
    "GW 33 (Data GW 1-32)": 32,
    "GW 36 (Data GW 1-35)": 35,
}

PROCESSED_SUBPATH  = os.path.join("processed", "musim 2025-2026")
PROCESSED_FILENAME = "players_processed_20252026.csv"

MATCH_THRESHOLD = 75


def fix_encoding(text):
    if not isinstance(text, str):
        return text
    try:
        return text.encode('latin-1').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text

def normalize_name(name):
    if not isinstance(name, str):
        return ""
    name = fix_encoding(name)
    name = unicodedata.normalize('NFKD', name)
    name = ''.join(c for c in name if not unicodedata.combining(c))
    name = name.lower().strip()
    return re.sub(r'\s+', ' ', name)

def load_vaastav_gw(gw_number):
    path = os.path.join(VAASTAV_DIR, f"gw{gw_number}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df['name_normalized'] = df['name'].apply(normalize_name)
    return df


def compute_rolling_avg(processed_df, gw_cutoff, window_size):
    """
    Untuk setiap pemain di processed_df, hitung rata-rata total_points
    dari (gw_cutoff - window_size + 1) sampai gw_cutoff.
    """
    gw_range = range(gw_cutoff - window_size + 1, gw_cutoff + 1)

    frames = []
    for gw in gw_range:
        df_gw = load_vaastav_gw(gw)
        if df_gw is None:
            print(f"    ⚠ GW{gw} tidak ditemukan, dilewati dari window")
            continue
        frames.append(df_gw[['name', 'name_normalized', 'total_points']])

    if not frames:
        print(f"    ✗ Tidak ada data GW dalam window {list(gw_range)}")
        processed_df['captain_metric'] = 0.0
        return processed_df

    df_window = pd.concat(frames, ignore_index=True)

    # Rata-rata per pemain (name_normalized), isi 0 kalau tidak main di GW itu
    avg_points = df_window.groupby('name_normalized')['total_points'].mean()

    vaastav_names = avg_points.index.tolist()

    processed_df = processed_df.copy()
    processed_df['name_normalized'] = processed_df.apply(
        lambda r: normalize_name(f"{r['first_name']} {r['second_name']}"), axis=1
    )

    captain_metrics = []
    for _, row in processed_df.iterrows():
        result = process.extractOne(
            row['name_normalized'], vaastav_names,
            scorer=fuzz.token_sort_ratio, score_cutoff=MATCH_THRESHOLD
        )
        captain_metrics.append(avg_points[result[0]] if result else 0.0)

    processed_df['captain_metric'] = captain_metrics
    return processed_df


def main():
    for folder_name, gw_cutoff in PERIOD_CUTOFF.items():
        print(f"\nPeriode: {folder_name} (window: GW{gw_cutoff-WINDOW_SIZE+1}-{gw_cutoff})")

        processed_path = os.path.join(
            DATA_DIR, folder_name, PROCESSED_SUBPATH, PROCESSED_FILENAME
        )
        if not os.path.exists(processed_path):
            print(f"  ✗ File processed tidak ditemukan, dilewati")
            continue

        df = pd.read_csv(processed_path)
        df = compute_rolling_avg(df, gw_cutoff, WINDOW_SIZE)

        df.to_csv(processed_path, index=False)
        print(f"  ✓ Kolom 'captain_metric' ditambahkan ke {len(df)} pemain")

    print("\nSelesai! Kolom 'captain_metric' sudah ditambahkan ke semua "
          "players_processed_20252026.csv.")
    print("Selanjutnya jalankan entropy_export.py.")

if __name__ == "__main__":
    main()