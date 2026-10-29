# Rekomendasi skuad Fantasy Premier League dengan Entropy-TOPSIS dan ILP

Kode dan data untuk memberi skor pemain dengan Entropy-TOPSIS per posisi, memilih 15 pemain dengan Integer Linear Programming, lalu membandingkan hasilnya dengan Average Point (AP) manajer global dan dua baseline pada gameweek 13 sampai 38 musim 2025/2026.

## Isi folder

- `code/` berisi skrip dan notebook, dengan urutan menjalankan di bawah.
- `data/` berisi delapan snapshot data pemain (mentah, olahan, dan ranking) musim 2025/2026.
- `results/` berisi bobot Entropy, skuad terpilih, hasil simulasi, dan data pembanding (AP).

## Delapan snapshot

Setiap snapshot diambil dari endpoint bootstrap-static setelah gameweek terakhir periodenya selesai. Endpoint hanya menampilkan keadaan terkini, jadi snapshot tidak bisa diambil ulang dan karena itu disertakan.

| Periode | Tanggal pengambilan |
|---|---|
| GW 13 | 28 November 2025 |
| GW 16 | 13 Desember 2025 |
| GW 19 | 30 Desember 2025 |
| GW 22 | 17 Januari 2026 |
| GW 26 | 11 Februari 2026 |
| GW 30 | 14 Maret 2026 |
| GW 33 | 18 April 2026 |
| GW 36 | 9 Mei 2026 |

## Menjalankan

Alur ini diuji dari awal dengan Python 3.14 dan paket di `requirements.txt`. Gunakan Python 3 dengan paket di `requirements.txt` (`pip install -r requirements.txt`). Semua skrip membaca folder proyek dari variabel lingkungan `FPL_BASE_DIR`, dan memakai folder induk `code/` kalau tidak diisi. Jalankan dari folder `code/`.

1. `download_vaastav.py` mengunduh poin per gameweek ke `results/vaastav/`.
2. `preprocessing_data.ipynb` membuat tabel olahan tiap periode.
3. `hitung_metrik_captain.py` menambah metrik kapten (rata-rata poin tiga gameweek terakhir) ke tabel olahan. Skrip ini harus berjalan sebelum langkah 4.
4. `entropy_export.py` menghitung bobot Entropy dan skor TOPSIS, lalu menulis ranking dan `results/entropy_weights/`.
5. `tambah_skor_baseline.py` menambah skor Baseline 1 dan Baseline 2 ke file ranking.
6. `seleksi_tim.py` memilih skuad Entropy-TOPSIS dengan ILP, dan `simulasi_murni.py` menghitung poinnya.
7. `seleksi_tim_baseline.py` dan `simulasi_baseline.py` mengerjakan hal yang sama untuk ketiga metode.
8. `uji_ketidakpastian.py` menghitung selisih berpasangan dan interval kepercayaan terhadap AP dan antarmetode.
9. `uji_sensitivitas_penalty_saved.py` menguji pengaruh kriteria penyelamatan penalti pada ranking kiper.

Skrip `audit_leakage.py` dan `rekonsiliasi_snapshot.py` memeriksa kebocoran data dan kesesuaian snapshot dengan poin per gameweek.
