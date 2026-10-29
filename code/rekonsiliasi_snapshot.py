import os
import sys

import pandas as pd

# Folder proyek = induk dari folder code/. Dapat diganti lewat FPL_BASE_DIR.
BASE_DIR = os.environ.get(
    "FPL_BASE_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
DATA_DIR = os.path.join(BASE_DIR, "data")
VAASTAV_DIR = os.path.join(BASE_DIR, "results", "vaastav")
OUTPUT_SUMMARY = os.path.join(BASE_DIR, "results", "rekonsiliasi_snapshot.csv")
OUTPUT_DECREASE = os.path.join(BASE_DIR, "results", "rekonsiliasi_penurunan.csv")

RAW_SUBPATH = os.path.join("raw", "players_raw_20252026.csv")

# (label periode, folder snapshot, gameweek terakhir yang tercakup snapshot)
PERIODS = [
    ("GW13", "GW 13 (Data GW 1-12)", 12),
    ("GW16", "GW 16 (Data GW 1-15)", 15),
    ("GW19", "GW 19 (Data GW 1-18)", 18),
    ("GW22", "GW 22 (Data GW 1-21)", 21),
    ("GW26", "GW 26 (Data GW 1-25)", 25),
    ("GW30", "GW 30 (Data GW 1-29)", 29),
    ("GW33", "GW 33 (Data GW 1-32)", 32),
    ("GW36", "GW 36 (Data GW 1-35)", 35),
]


def load_snapshot(folder):
    """Snapshot mentah: indeks id, kolom code dan total_points."""
    path = os.path.join(DATA_DIR, folder, RAW_SUBPATH)
    return pd.read_csv(path, usecols=["id", "code", "total_points"]).set_index("id")


def points_by_gameweek(first_gw, last_gw):
    """total_points vaastav per pemain (element) dan gameweek; baris ganda dijumlahkan."""
    frames = []
    for gw in range(first_gw, last_gw + 1):
        path = os.path.join(VAASTAV_DIR, f"gw{gw}.csv")
        df = pd.read_csv(path, usecols=["element", "total_points"])
        per_player = df.groupby("element")["total_points"].sum()
        frames.append(per_player.rename(gw))
    return pd.concat(frames, axis=1).fillna(0).astype(int)


CAUSE_COLUMNS = {
    "kartu_kuning": "yellow_cards",
    "kartu_merah": "red_cards",
    "gol_bunuh_diri": "own_goals",
    "penalti_gagal": "penalties_missed",
}


def negative_point_causes(player_id, gameweeks):
    """Jumlah kartu, gol bunuh diri, dan penalti gagal pemain pada gameweek berpoin negatif."""
    totals = {name: 0 for name in CAUSE_COLUMNS}
    for gw in gameweeks:
        path = os.path.join(VAASTAV_DIR, f"gw{gw}.csv")
        df = pd.read_csv(path, usecols=["element", *CAUSE_COLUMNS.values()])
        rows = df[df["element"] == player_id]
        for name, column in CAUSE_COLUMNS.items():
            totals[name] += int(rows[column].sum())
    return totals


def reconcile_transition(prev_label, prev_folder, prev_last_gw, next_label, next_folder, next_last_gw):
    prev = load_snapshot(prev_folder)
    nxt = load_snapshot(next_folder)
    first_gw, last_gw = prev_last_gw + 1, next_last_gw
    per_gw = points_by_gameweek(first_gw, last_gw)
    expected = per_gw.sum(axis=1)

    common = prev.index.intersection(nxt.index)
    new_ids = nxt.index.difference(prev.index)
    removed_ids = prev.index.difference(nxt.index)

    delta = nxt.loc[common, "total_points"] - prev.loc[common, "total_points"]
    expected_common = expected.reindex(common).fillna(0).astype(int)
    diff = delta - expected_common

    # Pemain yang baru muncul di snapshot berikutnya dianggap bernilai 0 sebelumnya.
    diff_new = nxt.loc[new_ids, "total_points"] - expected.reindex(new_ids).fillna(0).astype(int)

    code_changed = int((prev.loc[common, "code"] != nxt.loc[common, "code"]).sum())

    decreases = []
    for pid in delta[delta < 0].index:
        negative_gws = [gw for gw, p in per_gw.loc[pid].items() if p < 0] if pid in per_gw.index else []
        negatives = [f"GW{gw}:{int(per_gw.loc[pid, gw])}" for gw in negative_gws]
        causes = negative_point_causes(pid, negative_gws)
        decreases.append({
            "id": pid,
            "code": int(prev.loc[pid, "code"]),
            "transisi": f"{prev_label}-{next_label}",
            "poin_awal": int(prev.loc[pid, "total_points"]),
            "poin_akhir": int(nxt.loc[pid, "total_points"]),
            "selisih_snapshot": int(delta.loc[pid]),
            "jumlah_poin_vaastav": int(expected_common.loc[pid]),
            "selisih_rekonsiliasi": int(diff.loc[pid]),
            "gw_dengan_poin_negatif": ";".join(negatives),
            **causes,
        })

    summary = {
        "transisi": f"{prev_label}-{next_label}",
        "snapshot_awal": prev_label,
        "snapshot_akhir": next_label,
        "gameweek_dibandingkan": f"GW{first_gw}-GW{last_gw}",
        "jumlah_pemain_dibandingkan": len(common),
        "jumlah_pemain_baru": len(new_ids),
        "jumlah_pemain_hilang": len(removed_ids),
        "jumlah_selisih": int((diff != 0).sum()) + int((diff_new != 0).sum()),
        "selisih_absolut_maksimum": int(max(diff.abs().max(), diff_new.abs().max() if len(diff_new) else 0)),
        "jumlah_perubahan_code": code_changed,
        "jumlah_penurunan_total_poin": len(decreases),
    }
    return summary, decreases


def main():
    summaries, all_decreases = [], []
    for (a_label, a_folder, a_last), (b_label, b_folder, b_last) in zip(PERIODS[:-1], PERIODS[1:]):
        summary, decreases = reconcile_transition(a_label, a_folder, a_last, b_label, b_folder, b_last)
        summaries.append(summary)
        all_decreases.extend(decreases)
        print(f"{summary['transisi']}: {summary['jumlah_pemain_dibandingkan']} pemain, "
              f"{summary['jumlah_pemain_baru']} baru, selisih = {summary['jumlah_selisih']}, "
              f"penurunan = {summary['jumlah_penurunan_total_poin']}")

    df_summary = pd.DataFrame(summaries)
    df_decrease = pd.DataFrame(all_decreases)
    df_summary.to_csv(OUTPUT_SUMMARY, index=False)
    df_decrease.to_csv(OUTPUT_DECREASE, index=False)

    total_diff = int(df_summary["jumlah_selisih"].sum())
    total_decrease = len(df_decrease)
    explained = int(((df_decrease["selisih_rekonsiliasi"] == 0) &
                     (df_decrease["gw_dengan_poin_negatif"] != "")).sum()) if total_decrease else 0
    print(f"\nTotal selisih rekonsiliasi: {total_diff}")
    print(f"Penurunan total_points: {total_decrease} kasus, {explained} cocok dengan poin negatif per gameweek")
    print(f"Disimpan: {OUTPUT_SUMMARY}\n          {OUTPUT_DECREASE}")
    return 0 if total_diff == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
