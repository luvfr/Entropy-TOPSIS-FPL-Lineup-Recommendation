import os
from dataclasses import dataclass
from math import comb

import numpy as np
import pandas as pd

# Folder proyek = induk dari folder code/. Dapat diganti lewat FPL_BASE_DIR.
BASE_DIR = os.environ.get(
    "FPL_BASE_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
INPUT_METHODS = os.path.join(
    BASE_DIR, "results", "simulation_baseline", "simulation_baseline_results.csv"
)
INPUT_AP = os.path.join(BASE_DIR, "results", "ap_per_gameweek.csv")
OUTPUT = os.path.join(BASE_DIR, "results", "uji_ketidakpastian.csv")

N_BOOTSTRAP = 10_000
SEED = 20260
CI_LEVEL = 0.95
EXPECTED_GAMEWEEKS = 26
EXPECTED_PERIODS = 8
HIGHLIGHT_GAMEWEEKS = (28, 29)

METHOD_COLUMNS = {
    "entropy_topsis": "entropy",
    "baseline1_total_points": "baseline1",
    "baseline2_equal_weight": "baseline2",
}

# (kolom A, kolom B, label); selisih = A dikurangi B
PAIRS = [
    ("entropy", "ap", "Entropy-TOPSIS vs AP"),
    ("baseline1", "ap", "Baseline 1 vs AP"),
    ("baseline2", "ap", "Baseline 2 vs AP"),
    ("entropy", "baseline1", "Entropy-TOPSIS vs Baseline 1"),
    ("entropy", "baseline2", "Entropy-TOPSIS vs Baseline 2"),
]


@dataclass(frozen=True)
class PairResult:
    label: str
    n_gameweek: int
    mean_diff: float
    sd_diff: float
    ci_low: float
    ci_high: float
    wins: int
    ties: int
    losses: int
    p_sign: float
    total_diff: float
    total_diff_highlight: float


def load_wide(methods_path: str, ap_path: str) -> pd.DataFrame:
    """Gabungkan poin tiap metode dan AP menjadi satu baris per gameweek."""
    methods = pd.read_csv(methods_path)
    ap = pd.read_csv(ap_path)

    unknown = set(methods["metode"]) - set(METHOD_COLUMNS)
    if unknown:
        raise ValueError(f"Metode tidak dikenal pada {methods_path}: {sorted(unknown)}")

    wide = (
        methods.pivot(index="gw", columns="metode", values="total_points")
        .rename(columns=METHOD_COLUMNS)
    )
    periods = methods.drop_duplicates("gw").set_index("gw")["periode"]
    wide = wide.join(periods).join(ap.set_index("gw")["average_point"].rename("ap"))
    wide = wide.sort_index().reset_index()

    if len(wide) != EXPECTED_GAMEWEEKS or wide.isna().any().any():
        raise ValueError("Data tidak lengkap: harus 26 gameweek tanpa nilai kosong.")
    if wide["periode"].nunique() != EXPECTED_PERIODS:
        raise ValueError("Jumlah periode harus 8.")
    return wide


def sign_test_p(wins: int, losses: int) -> float:
    """p dua sisi binomial eksak (p = 0,5), seri dikeluarkan."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n
    return min(1.0, 2 * tail)


def period_index_draws(n_periods: int, n_boot: int, seed: int) -> np.ndarray:
    """Indeks periode untuk setiap resampling; dipakai bersama oleh semua pasangan."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_periods, size=(n_boot, n_periods))


def bootstrap_means(diff: np.ndarray, period_codes: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """Rata-rata selisih per resampling: jumlah selisih dibagi jumlah gameweek periode terpilih."""
    n_periods = draws.shape[1]
    sums = np.bincount(period_codes, weights=diff, minlength=n_periods)
    counts = np.bincount(period_codes, minlength=n_periods)
    return sums[draws].sum(axis=1) / counts[draws].sum(axis=1)


def analyse_pair(
    wide: pd.DataFrame, col_a: str, col_b: str, label: str,
    period_codes: np.ndarray, draws: np.ndarray,
) -> PairResult:
    diff = (wide[col_a] - wide[col_b]).to_numpy(dtype=float)
    wins, losses = int((diff > 0).sum()), int((diff < 0).sum())
    ties = len(diff) - wins - losses
    boot = bootstrap_means(diff, period_codes, draws)
    alpha = (1 - CI_LEVEL) / 2 * 100
    low, high = np.percentile(boot, [alpha, 100 - alpha])
    return PairResult(
        label=label,
        n_gameweek=len(diff),
        mean_diff=float(diff.mean()),
        sd_diff=float(diff.std(ddof=1)),
        ci_low=float(low),
        ci_high=float(high),
        wins=wins,
        ties=ties,
        losses=losses,
        p_sign=sign_test_p(wins, losses),
        total_diff=float(diff.sum()),
        total_diff_highlight=float(diff[wide["gw"].isin(HIGHLIGHT_GAMEWEEKS).to_numpy()].sum()),
    )


def run(methods_path: str = INPUT_METHODS, ap_path: str = INPUT_AP) -> pd.DataFrame:
    wide = load_wide(methods_path, ap_path)
    period_codes = pd.factorize(wide["periode"], sort=False)[0]
    draws = period_index_draws(int(period_codes.max()) + 1, N_BOOTSTRAP, SEED)
    results = [
        analyse_pair(wide, a, b, label, period_codes, draws) for a, b, label in PAIRS
    ]
    return pd.DataFrame(
        {
            "pasangan": [r.label for r in results],
            "n_gameweek": [r.n_gameweek for r in results],
            "n_periode": EXPECTED_PERIODS,
            "rata_rata_selisih": [round(r.mean_diff, 2) for r in results],
            "sd_selisih": [round(r.sd_diff, 2) for r in results],
            "ik95_bawah": [round(r.ci_low, 2) for r in results],
            "ik95_atas": [round(r.ci_high, 2) for r in results],
            "ik_memuat_nol": [r.ci_low <= 0 <= r.ci_high for r in results],
            "menang": [r.wins for r in results],
            "seri": [r.ties for r in results],
            "kalah": [r.losses for r in results],
            "p_uji_tanda": [round(r.p_sign, 4) for r in results],
            "total_selisih": [round(r.total_diff, 0) for r in results],
            "selisih_gw28_gw29": [round(r.total_diff_highlight, 0) for r in results],
            "n_bootstrap": N_BOOTSTRAP,
            "seed": SEED,
        }
    )


def main() -> None:
    table = run()
    table.to_csv(OUTPUT, index=False, encoding="utf-8")
    print(table.to_string(index=False))
    print(f"\nKeluaran: {OUTPUT}")


if __name__ == "__main__":
    main()
