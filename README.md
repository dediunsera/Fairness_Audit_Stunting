# Audit fairness model skrining stunting (SKI 2023), Studi A

Tanggal run: 20 Oktober 2026 · Seed 42 · n = 84.206 balita (33.131 PSU, 970 strata) · prevalensi stunting tertimbang 21,98% · 121 prediktor bebas leakage (pipeline yang sama dengan artikel HIR).

## Apa yang diuji
Model HGB yang sudah terkalibrasi diaudit per kelompok wilayah (7 region, 38 provinsi), kabupaten tertinggal 3T (Perpres 63/2020; 61 dari 62 kabupaten ada di SKI, n = 9.330), kota/desa, kuintil indeks kualitas rumah, status jaminan kesehatan, umur, dan jenis kelamin. Setiap anak diprediksi oleh model yang tidak pernah melihat PSU-nya (cross-fitting 5 fold). Metrik dihitung dengan bobot survei, dan selang kepercayaan memakai bootstrap PSU di dalam strata. Tujuh skenario dibandingkan: M0 (status quo), unawareness, awareness, M1 (rekalibrasi per region), M2 (ambang per region dengan FNR setara), M3 (reweighting seimbang), dan M4 (multikalibrasi). Ada juga tiga analisis sumber kesenjangan: model per region, leave-one-region-out, dan domain classifier.

## Hasil utama (angka dari `results/`)
1. **Kalibrasi nasional baik, kalibrasi lokal tidak.** Intercept keseluruhan −0,004 (ECE 0,004). Per region rentangnya −0,129 (Sumatera, risiko dilebihkan) sampai +0,084 (Sulawesi, risiko diremehkan). Per provinsi heterogenitasnya besar (τ = 0,166; I² = 85%): 11 dari 38 provinsi punya intercept empirical-Bayes yang CI-nya tidak memuat 0. Yang terburuk Bali (−0,56) dan Aceh (+0,27).
2. **Diskriminasi lebih rendah di daerah paling terbebani.** AUROC Papua 0,625, 3T 0,638, Papua perdesaan 0,604. Bandingkan Bali–Nusa Tenggara 0,715 dan nasional 0,661.
3. **Satu ambang nasional menghasilkan FNR yang sangat timpang, dan arahnya berlawanan dengan intuisi.** Anak stunting yang lolos (FNR) 0,10 di 3T vs 0,37 di non-3T; 0,14 di Papua vs 0,44 di Jawa; 0,59 di kuintil rumah terbaik. Penyebabnya perbedaan prevalensi (base rate), bukan bias model. Konsekuensinya, rujukan di 3T mencapai 810 per 1.000 anak (FPR 0,77).
4. **Sumber masalah.** Region sangat berbeda dalam ruang fitur (domain classifier AUROC 0,90–0,97). Model per region justru lebih buruk (AUROC keseluruhan 0,640 vs 0,661), jadi pooling membantu. Saat dipindah ke region baru (LORO), diskriminasi relatif bertahan, tetapi kalibrasi rusak (Jawa +0,267, Sumatera −0,203). Komponen yang tidak bisa dipindahkan adalah kalibrasi.
5. **Mitigasi.** Awareness (menambah region dan 3T sebagai prediktor) menurunkan max |intercept| region dari 0,129 menjadi 0,040 tanpa kehilangan AUROC (0,661). M1 menurunkannya menjadi 0,064. M2 menyamakan FNR antar region (rentang 0,298 → 0,051; gap 3T 0,269 → 0,158), dengan biaya +16 rujukan per 1.000 dan FNR Papua naik dari 0,14 ke 0,31. Unawareness dan reweighting tidak membantu.

**Artinya:** dalam skrining stunting, "ketidakadilan" yang paling relevan adalah kalibrasi lokal (provinsi) dan beban rujukan, bukan selisih FNR. Selisih FNR muncul dengan sendirinya dari perbedaan prevalensi jika memakai satu ambang risiko. Rekomendasinya: sertakan wilayah dalam model atau rekalibrasi lokal, laporkan kalibrasi per provinsi, dan pilih ambang secara eksplisit berdasarkan kapasitas rujukan setiap daerah. Menyamakan FNR adalah pilihan normatif yang menggeser beban, bukan koreksi bias.

## Reproduksi
```
cd scripts
python 01_prepare_data.py        # sama dengan pipeline HIR (butuh data/ SKI xlsx + lenanthro.txt)
python 02_attributes.py          # region, 3T, kuintil rumah, jaminan kesehatan
python 03_crossfit_models.py     # ~14 menit (2 CPU)
python 04_fairness_metrics.py    # ~32 menit (bootstrap)
python 05_figures.py
```
File `*.pkl` adalah turunan microdata SKI berlisensi: jangan dibagikan publik.

## Isi folder
- `results/comparison.csv` (= `T_mitigation_summary.csv`): tabel pembanding skenario
- `results/metrics.csv` (= `T_group_metrics.csv`): semua metrik per skenario × kelompok, dengan CI 95%
- `results/T_gaps.csv`: kesenjangan (max − min) per atribut, dengan CI
- `results/T_province_*.csv`: provinsi, empirical Bayes, heterogenitas
- `results/T_attr_*.csv`: pemetaan 3T, indeks rumah, profil atribut
- `results/T_domain_classifier.csv`, `T_thresholds.csv`, `T_intersectional_M0.csv`
- `figures/Fig1–Fig6`: 300 dpi
- `manuscript/`: naskah BMC Medical Informatics and Decision Making
