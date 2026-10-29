import pandas as pd
import os

# Folder proyek = induk dari folder code/. Dapat diganti lewat FPL_BASE_DIR.
BASE_DIR = os.environ.get(
    "FPL_BASE_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_SUBPATH = os.path.join("raw", "players_raw_20252026.csv")

PERIODS = [
    ("GW 13 (Data GW 1-12)",  "GW13"),
    ("GW 16 (Data GW 1-15)",  "GW16"),
    ("GW 19 (Data GW 1-18)",  "GW19"),
    ("GW 22 (Data GW 1-21)",  "GW22"),
    ("GW 26 (Data GW 1-25)",  "GW26"),
    ("GW 30 (Data GW 1-29)",  "GW30"),
    ("GW 33 (Data GW 1-32)",  "GW33"),
    ("GW 36 (Data GW 1-35)",  "GW36"),
]

def load_period(folder_name):
    path = os.path.join(DATA_DIR, folder_name, RAW_SUBPATH)
    df = pd.read_csv(path)
    # id unik pakai kolom 'code' (id resmi FPL, konsisten antar periode)
    return df.set_index('code')['total_points']

def main():
    series_per_period = {}
    for folder_name, label in PERIODS:
        try:
            series_per_period[label] = load_period(folder_name)
            print(f"✓ {label}: {len(series_per_period[label])} pemain dimuat")
        except FileNotFoundError:
            print(f"✗ {label}: file tidak ditemukan, dilewati")

    labels = list(series_per_period.keys())
    total_checked = 0
    total_violations = 0
    violation_details = []

    print(f"\n{'='*65}")
    print("Cek monotonisitas total_points untuk semua pemain, semua periode:")
    print(f"{'='*65}")

    for i in range(len(labels) - 1):
        p1, p2 = labels[i], labels[i + 1]
        s1, s2 = series_per_period[p1], series_per_period[p2]

        common_ids = s1.index.intersection(s2.index)
        pairwise = pd.DataFrame({p1: s1.loc[common_ids], p2: s2.loc[common_ids]})

        violations = pairwise[pairwise[p2] < pairwise[p1]]
        n_checked = len(pairwise)
        n_violation = len(violations)

        total_checked += n_checked
        total_violations += n_violation

        status = "✓ tidak ada penurunan" if n_violation == 0 else f"⚠ {n_violation} penurunan"
        print(f"  {p1} → {p2}: {n_checked} pemain dicek, {status}")

        if n_violation > 0:
            for pid, row in violations.iterrows():
                violation_details.append({
                    "code": pid, "dari": p1, "ke": p2,
                    "poin_awal": row[p1], "poin_akhir": row[p2],
                })

    print(f"\n{'='*65}")
    print(f"RINGKASAN: {total_checked} pasangan data-poin dicek total")
    print(f"Pelanggaran ditemukan: {total_violations} "
          f"({100*total_violations/total_checked:.3f}%)")
    print(f"{'='*65}")

    if violation_details:
        df_v = pd.DataFrame(violation_details)
        df_v.to_csv("pelanggaran_monoton.csv", index=False)
        print("\nDetail pelanggaran disimpan ke pelanggaran_monoton.csv")
        print("→ Cek satu-satu: kemungkinan pemain pindah klub/liga, id "
              "berubah, atau kasus tepi lain yang perlu ditelusuri.")
    else:
        print("\n✓ Tidak ada pemain yang total_points-nya turun antarperiode.")

if __name__ == "__main__":
    main()