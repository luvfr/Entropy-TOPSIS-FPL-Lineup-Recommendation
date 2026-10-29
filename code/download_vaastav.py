import requests
import os
import time


# Folder proyek = induk dari folder code/. Dapat diganti lewat FPL_BASE_DIR.
BASE_DIR = os.environ.get(
    "FPL_BASE_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
OUTPUT_DIR  = os.path.join(BASE_DIR, "results", "vaastav")

GW_START = 10
GW_END   = 38

BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/{}/data/2025-26/gws/gw{}.csv"

# Versi (commit) repositori vaastav yang dipakai penelitian. Ganti lewat VAASTAV_REF.
VAASTAV_REF = os.environ.get("VAASTAV_REF", "d7b05b51d137bfefef4b692386d18c610975b9b9")

# Delay antar request (detik) — hindari rate limiting
REQUEST_DELAY = 0.5


def download_gw(gw_number, output_dir, session):
    """Download satu file GW dari vaastav."""
    url      = BASE_URL.format(VAASTAV_REF, gw_number)
    filepath = os.path.join(output_dir, f"gw{gw_number}.csv")

    # Skip kalau sudah ada
    if os.path.exists(filepath):
        print(f"  GW{gw_number:02d}: sudah ada, skip")
        return True

    try:
        response = session.get(url, timeout=15)

        if response.status_code == 200:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(response.text)
            n_rows = len(response.text.strip().split('\n')) - 1
            print(f"  GW{gw_number:02d}: ✓ {n_rows} pemain")
            return True

        elif response.status_code == 404:
            print(f"  GW{gw_number:02d}: ✗ File tidak ditemukan (404)")
            return False

        else:
            print(f"  GW{gw_number:02d}: ✗ Error {response.status_code}")
            return False

    except requests.exceptions.Timeout:
        print(f"  GW{gw_number:02d}: ✗ Timeout")
        return False
    except requests.exceptions.ConnectionError:
        print(f"  GW{gw_number:02d}: ✗ Koneksi gagal")
        return False
    except Exception as e:
        print(f"  GW{gw_number:02d}: ✗ Error: {e}")
        return False


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"{'='*55}")
    print(f"  Download Vaastav FPL 2025/26 — GW{GW_START} s/d GW{GW_END}")
    print(f"{'='*55}")
    print(f"  Output: {OUTPUT_DIR}\n")

    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (research purposes)'
    })

    success = 0
    failed  = []

    for gw in range(GW_START, GW_END + 1):
        result = download_gw(gw, OUTPUT_DIR, session)
        if result:
            success += 1
        else:
            failed.append(gw)
        time.sleep(REQUEST_DELAY)

    print(f"\n{'='*55}")
    print(f"  Selesai! {success}/{GW_END - GW_START + 1} file berhasil didownload")
    if failed:
        print(f"  Gagal: GW {failed}")
        print(f"  Coba jalankan ulang script untuk retry file yang gagal")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*55}")

    if success > 0:
        import pandas as pd
        sample_file = os.path.join(OUTPUT_DIR, f"gw{GW_START}.csv")
        if os.path.exists(sample_file):
            df = pd.read_csv(sample_file)
            print(f"\n  Preview kolom GW{GW_START}:")
            print(f"  {list(df.columns)}")
            print(f"\n  Sample 3 baris pertama:")
            print(df[['name', 'position', 'total_points', 'minutes']].head(3).to_string(index=False))

if __name__ == "__main__":
    main()