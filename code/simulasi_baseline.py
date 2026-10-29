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
SQUAD_DIR   = os.path.join(BASE_DIR, "results", "squad_selection_baseline")
VAASTAV_DIR = os.path.join(BASE_DIR, "results", "vaastav")
OUTPUT_DIR  = os.path.join(BASE_DIR, "results", "simulation_baseline")

METHODS = ["entropy_topsis", "baseline1_total_points", "baseline2_equal_weight"]

PERIODS = [
    ("GW13", [13, 14, 15]), ("GW16", [16, 17, 18]), ("GW19", [19, 20, 21]),
    ("GW22", [22, 23, 24, 25]), ("GW26", [26, 27, 28, 29]), ("GW30", [30, 31, 32]),
    ("GW33", [33, 34, 35]), ("GW36", [36, 37, 38]),
]

MATCH_THRESHOLD = 75

def fix_encoding(text):
    if not isinstance(text, str): return text
    try: return text.encode('latin-1').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError): return text

def normalize_name(name):
    if not isinstance(name, str): return ""
    name = fix_encoding(name)
    name = unicodedata.normalize('NFKD', name)
    name = ''.join(c for c in name if not unicodedata.combining(c))
    return re.sub(r'\s+', ' ', name.lower().strip())

def load_squad(method_name, period_label):
    filepath = os.path.join(SQUAD_DIR, method_name, f"{period_label}_squad.csv")
    if not os.path.exists(filepath):
        return None
    df = pd.read_csv(filepath)
    df['name_normalized'] = df['full_name'].apply(normalize_name)
    return df

def load_vaastav(gw_number):
    filepath = os.path.join(VAASTAV_DIR, f"gw{gw_number}.csv")
    if not os.path.exists(filepath):
        return None
    df = pd.read_csv(filepath)
    df['name_normalized'] = df['name'].apply(normalize_name)
    return df

def match_player(name_normalized, vaastav_df, threshold=MATCH_THRESHOLD):
    # Pada gameweek ganda, semua baris dengan nama ternormalisasi yang sama
    # dijumlahkan (total_points dan minutes).
    vaastav_names = vaastav_df['name_normalized'].tolist()
    result = process.extractOne(name_normalized, vaastav_names,
                                 scorer=fuzz.token_sort_ratio, score_cutoff=threshold)
    if result:
        rows = vaastav_df[vaastav_df['name_normalized'] == result[0]]
        combined = pd.Series({
            'name': rows['name'].iloc[0],
            'total_points': rows['total_points'].sum(),
            'minutes': rows['minutes'].sum(),
        })
        return combined, result[1]
    return None, 0

def calculate_gw_points(squad_df, vaastav_df, gw):
    starting = squad_df[squad_df['role'].isin(
        ['Starting', 'Starting (C)', 'Starting (VC)'])].copy()
    bench = squad_df[squad_df['role'] == 'Bench'].copy()
 
    captain_row   = squad_df[squad_df['role'] == 'Starting (C)']
    vice_capt_row = squad_df[squad_df['role'] == 'Starting (VC)']
    captain_name   = captain_row.iloc[0]['full_name'] if len(captain_row) > 0 else None
    vice_capt_name = vice_capt_row.iloc[0]['full_name'] if len(vice_capt_row) > 0 else None
 
    player_data = {}
    for _, player in squad_df.iterrows():
        name, name_norm = player['full_name'], player['name_normalized']
        vaastav_row, score = match_player(name_norm, vaastav_df)
        if vaastav_row is not None:
            player_data[name] = {'points': int(vaastav_row.get('total_points', 0)),
                                  'minutes': int(vaastav_row.get('minutes', 0))}
        else:
            player_data[name] = {'points': 0, 'minutes': 0}
 
    # Formasi starting XI sebelum substitusi
    current_formation = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
    slots = []
    for _, p in starting.iterrows():
        name, pos = p['full_name'], p['position']
        mins = player_data.get(name, {'minutes': 0})['minutes']
        plays = mins > 0
        slots.append({'name': name, 'position': pos, 'plays': plays, 'sub_in': None})
        if plays:
            current_formation[pos] += 1

    POSITION_ORDER = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    slots.sort(key=lambda s: POSITION_ORDER[s['position']])
    
    # Memakai 'rank_score', kolom yang sama untuk ketiga metode
    bench_sorted = bench.sort_values('rank_score', ascending=False)
    bench_used = set()
 
    for slot in slots:
        if slot['plays']:
            continue
        for _, bench_p in bench_sorted.iterrows():
            bn, bp = bench_p['full_name'], bench_p['position']
            if bn in bench_used:
                continue
            bmins = player_data.get(bn, {'minutes': 0})['minutes']
            if bmins == 0:
                continue
            if (bp == 'GK') != (slot['position'] == 'GK'):
                continue
            new_formation = dict(current_formation)
            new_formation[bp] += 1
            if (new_formation['GK'] == 1 and new_formation['DEF'] >= 3 and
                    new_formation['MID'] >= 2 and new_formation['FWD'] >= 1):
                slot['sub_in'] = bn
                bench_used.add(bn)
                current_formation[bp] += 1
                break
 
    total_points = 0
    captain_plays = player_data.get(captain_name, {}).get('minutes', 0) > 0
    vc_plays = player_data.get(vice_capt_name, {}).get('minutes', 0) > 0
 
    for slot in slots:
        if slot['plays']:
            pts = player_data[slot['name']]['points']
        elif slot['sub_in']:
            pts = player_data[slot['sub_in']]['points']
        else:
            pts = 0
 
        name_for_captain = slot['name']
        if name_for_captain == captain_name and captain_plays:
            pts *= 2
        elif name_for_captain == vice_capt_name and not captain_plays and vc_plays:
            pts *= 2
 
        total_points += pts
    
    if gw == 28:
            print(f"\n[GW28] Rincian per pemain:")
            for slot in slots:
                status = "MAIN" if slot['plays'] else f"DISUB -> {slot['sub_in']}" if slot['sub_in'] else "0 POIN (tanpa sub)"
                pts_used = player_data[slot['name']]['points'] if slot['plays'] else \
                           (player_data[slot['sub_in']]['points'] if slot['sub_in'] else 0)
                print(f"  {slot['name']:<30} ({slot['position']}) {status:<25} poin dipakai: {pts_used}")
            print(f"  Total: {total_points}")
    
    return total_points

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = []

    for method_name in METHODS:
        print(f"\n{'='*65}\nMETODE: {method_name}\n{'='*65}")
        for period_label, gw_list in PERIODS:
            squad_df = load_squad(method_name, period_label)
            if squad_df is None:
                print(f"  x {period_label}: squad tidak ditemukan")
                continue
            for gw in gw_list:
                vaastav_df = load_vaastav(gw)
                if vaastav_df is None:
                    continue
                total_pts = calculate_gw_points(squad_df, vaastav_df, gw)
                print(f"  GW{gw}: {total_pts} poin")
                all_results.append({'metode': method_name, 'periode': period_label,
                                     'gw': gw, 'total_points': total_pts})

    df_results = pd.DataFrame(all_results)
    out_path = os.path.join(OUTPUT_DIR, "simulation_baseline_results.csv")
    df_results.to_csv(out_path, index=False)

    print(f"\n{'='*65}\nRINGKASAN PERBANDINGAN 3 METODE (26 GW)\n{'='*65}")
    summary = df_results.groupby('metode')['total_points'].agg(
        Total='sum', Rata_rata='mean').round(2)
    print(summary)
    print(f"\nDisimpan ke {out_path}")
    print("\nCATATAN: baris 'entropy_topsis' harus sama dengan hasil simulasi_murni.py "
          "(1326 total, 51.0 rata-rata).")

if __name__ == "__main__":
    main()