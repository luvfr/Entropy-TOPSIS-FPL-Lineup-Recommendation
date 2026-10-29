import pandas as pd
import numpy as np
import os
import sys
import pulp


# Folder proyek = induk dari folder code/. Dapat diganti lewat FPL_BASE_DIR.
BASE_DIR = os.environ.get(
    "FPL_BASE_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "squad_selection")

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

RANKING_SUBPATH      = os.path.join("output", "musim 2025-2026")
RANKING_FILES        = {
    "GK":  "GK_ranking_20252026.csv",
    "DEF": "DEF_ranking_20252026.csv",
    "MID": "MID_ranking_20252026.csv",
    "FWD": "FWD_ranking_20252026.csv",
}

SQUAD_COMPOSITION    = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
BUDGET               = 100.0
MAX_PER_CLUB         = 3
MIN_MINUTES_STARTING = 270
MIN_MINUTES_BENCH    = 90
VALID_FORMATIONS     = [
    (3, 4, 3), (3, 5, 2),
    (4, 3, 3), (4, 4, 2), (4, 5, 1),
    (5, 2, 3), (5, 3, 2), (5, 4, 1),
]


def fix_encoding(text):
    if not isinstance(text, str):
        return text
    try:
        return text.encode('latin-1').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text

def clean_name(first, second):
    return f"{fix_encoding(str(first)).strip()} {fix_encoding(str(second)).strip()}"


def load_rankings(folder_path):
    dfs = []
    for pos, filename in RANKING_FILES.items():
        filepath = os.path.join(folder_path, RANKING_SUBPATH, filename)
        if not os.path.exists(filepath):
            print(f"  ⚠ File tidak ditemukan: {filepath}")
            continue
        df = pd.read_csv(filepath)
        df['position'] = pos
        df['full_name'] = df.apply(
            lambda r: clean_name(r['first_name'], r['second_name']), axis=1
        )
        dfs.append(df)
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)

def add_position_normalized_score(df_all, score_col='topsis_score',
                                    new_col='position_norm_score'):
    """Percentile rank (0-1) skor DI DALAM tiap posisi, untuk
    keputusan lintas posisi (ILP objective, pemilihan formasi)."""
    df_all = df_all.copy()
    df_all[new_col] = df_all.groupby('position')[score_col].rank(pct=True)
    return df_all


def require_team_column(df):
    """Batasan klub wajib berjalan: hentikan proses bila kolom 'team' tidak ada."""
    if 'team' not in df.columns:
        sys.exit("KESALAHAN: kolom 'team' tidak ada pada data ranking. "
                 "Batasan maksimal 3 pemain per klub tidak dapat diterapkan. "
                 "Jalankan ulang preprocessing dengan 'team' pada desired_columns.")
    if df['team'].isna().any():
        sys.exit("KESALAHAN: kolom 'team' memuat nilai kosong.")

def select_squad_ilp(df_all, budget=100.0):
    require_team_column(df_all)
    has_minutes = 'minutes' in df_all.columns

    if has_minutes:
        df = df_all[df_all['minutes'] >= MIN_MINUTES_BENCH].copy()
        n_filtered = len(df_all) - len(df)
        print(f"  ✓ Eligible (>={MIN_MINUTES_BENCH} menit): "
              f"{len(df)} pemain ({n_filtered} dikeluarkan)")
    else:
        df = df_all.copy()
        print(f"  ⚠ Kolom 'minutes' tidak ditemukan, filter menit dinonaktifkan")

    df = df.reset_index(drop=True)
    n  = len(df)

    prob = pulp.LpProblem("FPL_Squad_Selection", pulp.LpMaximize)

    x = [pulp.LpVariable(f"x_{i}", cat='Binary') for i in range(n)]

    prob += pulp.lpSum(df.loc[i, 'position_norm_score'] * x[i] for i in range(n))

    prob += pulp.lpSum(df.loc[i, 'price'] * x[i] for i in range(n)) <= budget

    for pos, count in SQUAD_COMPOSITION.items():
        idx_pos = df[df['position'] == pos].index.tolist()
        prob += pulp.lpSum(x[i] for i in idx_pos) == count

    for club in df['team'].unique():
        idx_club = df[df['team'] == club].index.tolist()
        prob += pulp.lpSum(x[i] for i in idx_club) <= MAX_PER_CLUB

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] != 'Optimal':
        print(f"  ✗ ILP tidak menemukan solusi optimal: {pulp.LpStatus[prob.status]}")
        return pd.DataFrame(), budget

    selected_idx = [i for i in range(n) if pulp.value(x[i]) == 1]
    squad_df     = df.loc[selected_idx].copy()
    total_price  = squad_df['price'].sum()
    sisa_budget  = budget - total_price

    print(f"  ✓ ILP status: Optimal | {len(squad_df)} pemain | "
          f"total: {total_price:.1f}M | sisa: {sisa_budget:.1f}M")

    return squad_df, sisa_budget

def check_club_limit(squad_df, label):
    """Hitung pemain per klub, cetak, dan hentikan bila ada klub melebihi batas."""
    counts = squad_df.groupby('team').size().sort_values(ascending=False)
    detail = ', '.join(f"{club}: {n}" for club, n in counts.items())
    print(f"  ✓ Pemain per klub (maks {counts.max()}): {detail}")
    if counts.max() > MAX_PER_CLUB:
        sys.exit(f"KESALAHAN: {label} melanggar batas {MAX_PER_CLUB} pemain per klub "
                 f"(klub {counts.index[0]}: {counts.iloc[0]} pemain).")
    return counts


def select_starting_xi(squad_df):
    """
    Pilih starting XI yang memaksimalkan total skor normalisasi.
    GK starting selalu yang skor normalisasi tertinggi (post-processing swap).
    """
    best_score, best_starting, best_bench, best_formation = -1, None, None, None

    # GK selalu dipilih berdasarkan skor normalisasi tertinggi
    gk_pool     = squad_df[squad_df['position'] == 'GK'].sort_values(
        'position_norm_score', ascending=False)
    starting_gk = gk_pool.iloc[[0]]
    bench_gk    = gk_pool.iloc[1:]

    defs = squad_df[squad_df['position'] == 'DEF'].sort_values(
        'position_norm_score', ascending=False)
    mids = squad_df[squad_df['position'] == 'MID'].sort_values(
        'position_norm_score', ascending=False)
    fwds = squad_df[squad_df['position'] == 'FWD'].sort_values(
        'position_norm_score', ascending=False)

    for n_def, n_mid, n_fwd in VALID_FORMATIONS:
        if n_def > len(defs) or n_mid > len(mids) or n_fwd > len(fwds):
            continue

        starting = pd.concat([
            starting_gk,
            defs.iloc[:n_def],
            mids.iloc[:n_mid],
            fwds.iloc[:n_fwd]
        ])
        bench = pd.concat([
            bench_gk,
            defs.iloc[n_def:],
            mids.iloc[n_mid:],
            fwds.iloc[n_fwd:]
        ])

        score = starting['position_norm_score'].sum()
        if score > best_score:
            best_score     = score
            best_starting  = starting
            best_bench     = bench
            best_formation = f"1-{n_def}-{n_mid}-{n_fwd}"

    return best_starting, best_bench, best_formation


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"{'='*65}")
    print(f"  Seleksi Skuad (ILP) — Entropy-TOPSIS FPL 2025/26")
    print(f"  Starting threshold: {MIN_MINUTES_STARTING} menit")
    print(f"  Bench threshold   : {MIN_MINUTES_BENCH} menit")
    print(f"{'='*65}\n")

    summary_rows = []

    for folder_name, period_label in PERIODS:
        folder_path = os.path.join(DATA_DIR, folder_name)
        print(f"{'─'*65}")
        print(f"  Periode: {period_label} ({folder_name})")

        if not os.path.exists(folder_path):
            print(f"  ✗ Folder tidak ditemukan\n")
            continue

        df_all = load_rankings(folder_path)
        if df_all is None or df_all.empty:
            print(f"  ✗ Tidak ada data ranking\n")
            continue
        
        df_all = add_position_normalized_score(df_all)
        print(f"  ✓ {len(df_all)} pemain dimuat")

        squad_df, sisa_budget = select_squad_ilp(df_all, BUDGET)
        if squad_df.empty:
            continue
        check_club_limit(squad_df, period_label)

        starting_df, bench_df, formation = select_starting_xi(squad_df)
        print(f"  ✓ Formasi optimal: {formation}")

        starting_sorted = starting_df.sort_values('captain_metric', ascending=False)
        captain   = starting_sorted.iloc[0]['full_name']
        vice_capt = starting_sorted.iloc[1]['full_name']
        print(f"  ✓ Captain     : {captain}")
        print(f"  ✓ Vice Captain: {vice_capt}")

        squad_df = squad_df.copy()
        squad_df['role'] = squad_df['full_name'].apply(
            lambda n: 'Starting' if n in starting_df['full_name'].values else 'Bench'
        )
        squad_df.loc[squad_df['full_name'] == captain,   'role'] = 'Starting (C)'
        squad_df.loc[squad_df['full_name'] == vice_capt, 'role'] = 'Starting (VC)'
        squad_df['period']    = period_label
        squad_df['formation'] = formation

        if 'minutes' in squad_df.columns:
            squad_df['tier'] = squad_df['minutes'].apply(
                lambda m: 'T1' if m >= MIN_MINUTES_STARTING else 'T2'
            )

        print(f"\n  {'Nama':<32} {'Pos':<5} {'Harga':>7} {'TOPSIS':>8} "
              f"{'Menit':>7}  Role")
        print(f"  {'─'*70}")
        for _, p in squad_df.sort_values(
            ['role', 'position', 'topsis_score'],
            ascending=[True, True, False]
        ).iterrows():
            tag  = {"Starting (C)": "(C)", "Starting (VC)": "(VC)",
                    "Bench": "[Bench]"}.get(p['role'], "")
            mins = int(p['minutes']) if 'minutes' in p \
                   and not pd.isna(p['minutes']) else 0
            tier = p.get('tier', '')
            print(f"  {p['full_name']:<32} {p['position']:<5} "
                  f"{p['price']:>7.1f} {p['topsis_score']:>8.4f} "
                  f"{mins:>7}  {tag} {tier}")

        cols_save = ['period', 'full_name', 'position', 'team', 'price', 'minutes',
                     'topsis_score', 'position_norm_score', 'rank', 'role', 'tier', 'formation']
        cols_save = [c for c in cols_save if c in squad_df.columns]
        squad_df[cols_save].sort_values(
            ['position', 'topsis_score'], ascending=[True, False]
        ).to_csv(os.path.join(OUTPUT_DIR, f"{period_label}_squad.csv"), index=False)

        summary_rows.append({
            'Periode':          folder_name,
            'Formasi':          formation,
            'Captain':          captain,
            'Vice Captain':     vice_capt,
            'GK':  ', '.join(
                squad_df[squad_df['position']=='GK']['full_name'].tolist()),
            'DEF': ', '.join(
                squad_df[squad_df['position']=='DEF']['full_name'].tolist()),
            'MID': ', '.join(
                squad_df[squad_df['position']=='MID']['full_name'].tolist()),
            'FWD': ', '.join(
                squad_df[squad_df['position']=='FWD']['full_name'].tolist()),
            'Total Budget (M)': round(squad_df['price'].sum(), 1),
            'Sisa Budget (M)':  round(sisa_budget, 1),
        })
        print()

    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(
            os.path.join(OUTPUT_DIR, "summary_squad_all_periods.csv"),
            index=False
        )
        print(f"{'─'*65}")
        print("  Ringkasan semua periode:")
        print(df_summary[
            ['Periode', 'Formasi', 'Captain',
             'Total Budget (M)', 'Sisa Budget (M)']
        ].to_string(index=False))

    print(f"\n{'='*65}")
    print(f"  Selesai! Output: {OUTPUT_DIR}")
    print(f"{'='*65}")

if __name__ == "__main__":
    main()