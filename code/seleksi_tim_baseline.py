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
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "squad_selection_baseline")

PERIODS = [
    ("GW 13 (Data GW 1-12)",  "GW13"), ("GW 16 (Data GW 1-15)",  "GW16"),
    ("GW 19 (Data GW 1-18)",  "GW19"), ("GW 22 (Data GW 1-21)",  "GW22"),
    ("GW 26 (Data GW 1-25)",  "GW26"), ("GW 30 (Data GW 1-29)",  "GW30"),
    ("GW 33 (Data GW 1-32)",  "GW33"), ("GW 36 (Data GW 1-35)",  "GW36"),
]

RANKING_SUBPATH = os.path.join("output", "musim 2025-2026")
RANKING_FILES = {
    "GK": "GK_ranking_20252026.csv", "DEF": "DEF_ranking_20252026.csv",
    "MID": "MID_ranking_20252026.csv", "FWD": "FWD_ranking_20252026.csv",
}

SQUAD_COMPOSITION    = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
BUDGET               = 100.0
MAX_PER_CLUB         = 3
MIN_MINUTES_BENCH    = 90
VALID_FORMATIONS     = [
    (3, 4, 3), (3, 5, 2), (4, 3, 3), (4, 4, 2), (4, 5, 1),
    (5, 2, 3), (5, 3, 2), (5, 4, 1),
]

# (nama_metode, nama_kolom_skor)
METHODS = [
    ("entropy_topsis", "topsis_score"),
    ("baseline1_total_points", "baseline1_score"),
    ("baseline2_equal_weight", "baseline2_score"),
]

def fix_encoding(text):
    if not isinstance(text, str): return text
    try: return text.encode('latin-1').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError): return text

def clean_name(first, second):
    return f"{fix_encoding(str(first)).strip()} {fix_encoding(str(second)).strip()}"

def load_rankings(folder_path):
    dfs = []
    for pos, filename in RANKING_FILES.items():
        filepath = os.path.join(folder_path, RANKING_SUBPATH, filename)
        if not os.path.exists(filepath):
            continue
        df = pd.read_csv(filepath)
        df['position'] = pos
        df['full_name'] = df.apply(lambda r: clean_name(r['first_name'], r['second_name']), axis=1)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else None

def add_position_normalized_score(df_all, score_col, new_col='position_norm_score'):
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

def select_squad_ilp(df_all, score_col, budget=100.0):
    require_team_column(df_all)
    if 'minutes' in df_all.columns:
        df = df_all[df_all['minutes'] >= MIN_MINUTES_BENCH].copy()
    else:
        df = df_all.copy()
    df = df.reset_index(drop=True)
    n = len(df)

    prob = pulp.LpProblem("Squad_Selection", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x_{i}", cat='Binary') for i in range(n)]
    prob += pulp.lpSum(df.loc[i, score_col] * x[i] for i in range(n))
    prob += pulp.lpSum(df.loc[i, 'price'] * x[i] for i in range(n)) <= budget
    for pos, count in SQUAD_COMPOSITION.items():
        idx_pos = df[df['position'] == pos].index.tolist()
        prob += pulp.lpSum(x[i] for i in idx_pos) == count
    for club in df['team'].unique():
        idx_club = df[df['team'] == club].index.tolist()
        prob += pulp.lpSum(x[i] for i in idx_club) <= MAX_PER_CLUB

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] != 'Optimal':
        return pd.DataFrame(), budget

    selected_idx = [i for i in range(n) if pulp.value(x[i]) == 1]
    squad_df = df.loc[selected_idx].copy()
    sisa_budget = budget - squad_df['price'].sum()
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

def select_starting_xi(squad_df, score_col):
    gk_pool = squad_df[squad_df['position'] == 'GK'].sort_values(score_col, ascending=False)
    starting_gk, bench_gk = gk_pool.iloc[[0]], gk_pool.iloc[1:]
    defs = squad_df[squad_df['position'] == 'DEF'].sort_values(score_col, ascending=False)
    mids = squad_df[squad_df['position'] == 'MID'].sort_values(score_col, ascending=False)
    fwds = squad_df[squad_df['position'] == 'FWD'].sort_values(score_col, ascending=False)

    best_score, best_starting, best_bench, best_formation = -1, None, None, None
    for n_def, n_mid, n_fwd in VALID_FORMATIONS:
        if n_def > len(defs) or n_mid > len(mids) or n_fwd > len(fwds):
            continue
        starting = pd.concat([starting_gk, defs.iloc[:n_def], mids.iloc[:n_mid], fwds.iloc[:n_fwd]])
        bench = pd.concat([bench_gk, defs.iloc[n_def:], mids.iloc[n_mid:], fwds.iloc[n_fwd:]])
        score = starting[score_col].sum()
        if score > best_score:
            best_score, best_starting, best_bench = score, starting, bench
            best_formation = f"1-{n_def}-{n_mid}-{n_fwd}"
    return best_starting, best_bench, best_formation

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for method_name, score_col in METHODS:
        print(f"\n{'='*65}\nMETODE: {method_name} (kolom skor: {score_col})\n{'='*65}")
        method_dir = os.path.join(OUTPUT_DIR, method_name)
        os.makedirs(method_dir, exist_ok=True)

        for folder_name, period_label in PERIODS:
            folder_path = os.path.join(DATA_DIR, folder_name)
            df_all = load_rankings(folder_path)
            if df_all is None or df_all.empty:
                print(f"  x {period_label}: data tidak ditemukan")
                continue
            
            df_all = add_position_normalized_score(df_all, score_col)

            squad_df, sisa_budget = select_squad_ilp(df_all, 'position_norm_score', BUDGET)
            if squad_df.empty:
                print(f"  x {period_label}: ILP gagal menemukan solusi")
                continue
            check_club_limit(squad_df, f"{method_name} {period_label}")

            starting_df, bench_df, formation = select_starting_xi(squad_df, 'position_norm_score')

            squad_df = squad_df.copy()
            squad_df['role'] = squad_df['full_name'].apply(
                lambda n: 'Starting' if n in starting_df['full_name'].values else 'Bench')

            # Kapten dipilih dengan captain_metric yang sama untuk semua metode
            starting_sorted = starting_df.sort_values('captain_metric', ascending=False)
            captain = starting_sorted.iloc[0]['full_name']
            vice_capt = starting_sorted.iloc[1]['full_name']
            squad_df.loc[squad_df['full_name'] == captain,   'role'] = 'Starting (C)'
            squad_df.loc[squad_df['full_name'] == vice_capt, 'role'] = 'Starting (VC)'
            squad_df['period']     = period_label
            squad_df['formation']  = formation
            # samakan nama kolom skor supaya konsisten dibaca simulasi_baseline.py
            squad_df['rank_score'] = squad_df['position_norm_score']

            cols_save = ['period', 'full_name', 'position', 'team', 'price', 'minutes',
                         'rank_score', 'captain_metric', 'role', 'formation']
            cols_save = [c for c in cols_save if c in squad_df.columns]
            squad_df[cols_save].to_csv(
                os.path.join(method_dir, f"{period_label}_squad.csv"), index=False)

            print(f"  v {period_label}: formasi {formation} | kapten {captain} | "
                  f"budget {BUDGET - sisa_budget:.1f}M")

    print(f"\nSelesai! Selanjutnya jalankan simulasi_baseline.py")

if __name__ == "__main__":
    main()