# Laporan Perbandingan MP3 vs WAV – BeatNusa

## 1. Informasi Pengujian

| Parameter | Nilai |
|---|---|
| Tanggal dan waktu | 2026-07-03 16:12:23 |
| Sistem operasi | Windows 10 10.0.26200 |
| CPU | 12th Gen Intel(R) Core(TM) i5-12500H |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| CUDA tersedia | True |
| Device digunakan | cuda |
| Versi Python | 3.11.0rc2 |
| Versi PyTorch | 2.5.1+cu121 |
| Versi Librosa | 0.11.0 |
| Path checkpoint | D:\Tugas\Semester 7\Skripsi\beatmap_bert_project_real - Copy\checkpoints\best.pt |
| Epoch checkpoint | 30 |
| Path konfigurasi | D:\Tugas\Semester 7\Skripsi\beatmap_bert_project_real - Copy\configs\local.yaml |
| Jumlah pengulangan | 3 |

## 2. Validasi Kesamaan Audio

- Durasi MP3: 107.2994 detik
- Durasi WAV: 107.2994 detik
- Selisih durasi: 0.00 ms
- ✅ Selisih durasi ≤ 100 ms. Kedua file kemungkinan berasal dari lagu yang sama.
- Pearson correlation spectrogram: 0.9997767806053162
- Cosine similarity spectrogram: 0.9997784495353699

## 3. Perbandingan Karakteristik File

| Karakteristik | MP3 | WAV |
|---|---|---|
| filename | soleramlagudaerahriaubyviolamerliaqdaaf.mp3 | soleramlagudaerahriaubyviolamerliaqdaaf.wav |
| format | MP3 | WAV |
| size_bytes | 1984773 | 20601566 |
| size_mb | 1.8928 | 19.6472 |
| duration_seconds | 107.29941666666667 | 107.29941666666667 |
| sample_rate | 48000 | 48000 |
| channels | 2 | 2 |
| codec | MP3 | PCM_16 |
| bitrate_kbps | 148 | tidak tersedia (WAV) |
| bit_depth | tidak tersedia (MP3) | PCM_16 |

### Penanganan File oleh Backend

**MP3:**

- extension: .mp3
- is_allowed_format: True
- file_size_bytes: 1984773
- max_upload_bytes: 10485760
- size_within_limit: True
- validation_passed: True
- original_format: mp3
- transcoded: True
- transcoded_format: wav
- transcoded_path: C:\Users\hp\AppData\Local\Temp\beatnusa_cmp_gi_yx28x\soleramlagudaerahriaubyviolamerliaqdaaf_transcoded.wav
- transcoded_size_bytes: 20601532
- transcoding_time_s: 0.17871020000075077
- before_sr: 48000
- before_channels: 2
- after_sr: 48000
- after_channels: 2
- after_subtype: PCM_16

**WAV:**

- extension: .wav
- is_allowed_format: True
- file_size_bytes: 20601566
- max_upload_bytes: 10485760
- size_within_limit: False
- validation_passed: False
- original_format: wav
- transcoded: False
- transcoded_path: C:\Users\hp\AppData\Local\Temp\beatnusa_cmp_gi_yx28x\backend_soleramlagudaerahriaubyviolamerliaqdaaf.wav
- handling_time_s: 0.012463600000046426
- note: File sudah WAV, hanya di-copy (rename di backend asli)
- after_sr: 48000
- after_channels: 2
- after_subtype: PCM_16

## 4. Perbandingan Waktu Pemrosesan

| Tahap | MP3 Rata-rata (s) | MP3 Std | MP3 Min | MP3 Max | WAV Rata-rata (s) | WAV Std | WAV Min | WAV Max |
|---|---|---|---|---|---|---|---|---|
| Audio Load | 0.157522 | 0.021900 | 0.141887 | 0.188493 | 0.127752 | 0.006727 | 0.119293 | 0.135751 |
| Melpectrogram | 0.284223 | 0.010076 | 0.275639 | 0.298364 | 0.265485 | 0.023320 | 0.241341 | 0.297014 |
| Rhythm Guides | 0.779884 | 0.051440 | 0.708267 | 0.826753 | 0.739314 | 0.009297 | 0.732452 | 0.752457 |
| Preprocessing Total | 1.221629 | 0.076596 | 1.126090 | 1.313611 | 1.132551 | 0.038757 | 1.093086 | 1.185223 |
| Inference | 0.128463 | 0.001067 | 0.127619 | 0.129968 | 0.127961 | 0.000395 | 0.127444 | 0.128403 |
| Postprocessing | 0.002772 | 0.000322 | 0.002404 | 0.003189 | 0.003089 | 0.000293 | 0.002731 | 0.003450 |
| Beatmap Json Build | 0.000062 | 0.000007 | 0.000054 | 0.000071 | 0.000083 | 0.000009 | 0.000076 | 0.000096 |
| Pipeline Total | 1.357011 | 0.076969 | 1.260412 | 1.448760 | 1.267890 | 0.038551 | 1.228794 | 1.320345 |

## 5. Perbandingan Hasil Preprocessing

| Parameter | MP3 | WAV |
|---|---|---|
| waveform_sr | 22050 | 22050 |
| waveform_channels | 1 | 1 |
| waveform_samples | 2365953 | 2365953 |
| waveform_duration_s | 107.299456 | 107.299456 |
| mel_shape | [128, 9243] | [128, 9243] |
| mel_time_frames | 9243 | 9243 |
| mel_min_pre_norm | -80.000000 | -80.000000 |
| mel_max_pre_norm | 0.000000 | 0.000000 |
| mel_pre_norm_mean | -39.231762 | -39.208344 |
| mel_pre_norm_std | 13.019136 | 13.025229 |
| mel_min_post_norm | -3.131409 | -3.131742 |
| mel_max_post_norm | 3.013392 | 3.010185 |
| mel_post_norm_mean | -0.000000 | -0.000000 |
| mel_post_norm_std | 1.000000 | 1.000000 |
| num_onsets | 370 | 370 |
| num_beats | 177 | 177 |
| bpm | 105.468750 | 105.468750 |

### Kemiripan Spectrogram

| Metrik | Nilai |
|---|---|
| frames_compared | 9243 |
| extra_frames_mp3 | 0 |
| extra_frames_wav | 0 |
| mae | 0.01363855 |
| rmse | 0.02082808 |
| pearson_correlation | 0.99977678 |
| cosine_similarity | 0.99977845 |

## 6. Perbandingan Prediksi Mentah

| Parameter | MP3 | WAV |
|---|---|---|
| pred_frames | 9243 | 9243 |
| event_prob_min | 0.035390 | 0.035224 |
| event_prob_max | 0.805705 | 0.805025 |
| event_prob_mean | 0.229957 | 0.230079 |
| event_prob_std | 0.133785 | 0.133768 |
| frames_above_eval_threshold | 879 | 874 |
| frames_above_pp_threshold | 291 | 291 |
| num_candidates_pre_pp | 291 | 291 |

### Distribusi Lane Sebelum Post-Processing

| Lane | MP3 | WAV |
|---|---|---|
| Lane 0 | 15 | 17 |
| Lane 1 | 237 | 234 |
| Lane 2 | 38 | 39 |
| Lane 3 | 1 | 1 |

### Contoh Frame Prediksi (5 frame pertama dengan probabilitas tertinggi)

**MP3:**

| Frame | Probabilitas Event | Lane Prediksi |
|---|---|---|
| 5511 | 0.805705 | 2 |
| 7135 | 0.797688 | 1 |
| 3149 | 0.784281 | 1 |
| 8217 | 0.781926 | 1 |
| 5806 | 0.781363 | 1 |

**WAV:**

| Frame | Probabilitas Event | Lane Prediksi |
|---|---|---|
| 5511 | 0.805025 | 2 |
| 7135 | 0.792998 | 1 |
| 3149 | 0.783253 | 1 |
| 8217 | 0.782644 | 1 |
| 5806 | 0.780273 | 1 |

## 7. Perbandingan Beatmap Akhir

| Parameter | MP3 | WAV |
|---|---|---|
| pp_event_threshold | 0.6 | 0.6 |
| pp_min_gap_ms | 200 | 200 |
| pp_same_lane_min_gap_ms | 250 | 250 |
| pp_onset_snap_tolerance_ms | 30 | 30 |
| pp_beat_snap_tolerance_ms | 40 | 40 |
| pp_max_density_nps | 3 | 3 |
| num_candidates_pre_pp | 291 | 291 |
| num_notes_after_pp | 216 | 216 |
| notes_removed | 75 | 75 |
| notes_removed_pct | 25.77 | 25.77 |
| beatmap_duration_ms | 107299.45578231291 | 107299.45578231291 |
| note_density_per_s | 2.0131 | 2.0131 |
| beatmap_json_size_bytes | 13494 | 13494 |
| beatmap_success | True | True |

### Distribusi Lane Setelah Post-Processing

| Lane | MP3 | WAV |
|---|---|---|
| Lane 0 | 14 | 16 |
| Lane 1 | 174 | 175 |
| Lane 2 | 27 | 24 |
| Lane 3 | 1 | 1 |

## 8. Kemiripan Beatmap MP3 dan WAV

### Toleransi 25 ms

| Metrik | Waktu Saja | Waktu + Lane |
|---|---|---|
| Jumlah note MP3 | 216 | 216 |
| Jumlah note WAV | 216 | 216 |
| Note cocok | 209 | 206 |
| Note MP3 tanpa pasangan | 7 | 10 |
| Note WAV tanpa pasangan | 7 | 10 |
| Rata-rata selisih timestamp (ms) | 0.11483253588516747 | 0.11650485436893204 |
| Median selisih timestamp (ms) | 0.0 | 0.0 |
| Maks selisih timestamp (ms) | 24.0 | 24.0 |

### Toleransi 50 ms

| Metrik | Waktu Saja | Waktu + Lane |
|---|---|---|
| Jumlah note MP3 | 216 | 216 |
| Jumlah note WAV | 216 | 216 |
| Note cocok | 210 | 207 |
| Note MP3 tanpa pasangan | 6 | 9 |
| Note WAV tanpa pasangan | 6 | 9 |
| Rata-rata selisih timestamp (ms) | 0.28095238095238095 | 0.28502415458937197 |
| Median selisih timestamp (ms) | 0.0 | 0.0 |
| Maks selisih timestamp (ms) | 35.0 | 35.0 |

### Toleransi 100 ms

| Metrik | Waktu Saja | Waktu + Lane |
|---|---|---|
| Jumlah note MP3 | 216 | 216 |
| Jumlah note WAV | 216 | 216 |
| Note cocok | 210 | 207 |
| Note MP3 tanpa pasangan | 6 | 9 |
| Note WAV tanpa pasangan | 6 | 9 |
| Rata-rata selisih timestamp (ms) | 0.28095238095238095 | 0.28502415458937197 |
| Median selisih timestamp (ms) | 0.0 | 0.0 |
| Maks selisih timestamp (ms) | 35.0 | 35.0 |

**Tingkat kesesuaian lane** pada note yang cocok (100 ms): 98.57%

**Kemiripan distribusi lane** (cosine similarity): 0.999785

**Perbedaan kepadatan note**: 0.0000 notes/detik

**Jaccard similarity** (100 ms, waktu saja): 0.945946

## 9. Determinisme Hasil

- Note count konsisten antar-run (MP3): True
- Note count konsisten antar-run (WAV): True
- Timestamp konsisten antar-run (MP3): True
- Timestamp konsisten antar-run (WAV): True
- Lane konsisten antar-run (MP3): True
- Lane konsisten antar-run (WAV): True

✅ Semua hasil deterministik. Perubahan hanya terjadi pada waktu eksekusi.

## 10. Analisis

- **Format lebih kecil**: MP3 (1.8928 MB vs 19.6472 MB, rasio 9.6%)
- **Format lebih cepat diproses**: WAV (1.2679s vs 1.3570s)
- **Kemiripan preprocessing**: Pearson correlation spectrogram = 0.999777. MAE = 0.013639, RMSE = 0.020828
- **Kemiripan beatmap akhir** (50 ms): 210 note cocok dari 216 (MP3) dan 216 (WAV) = 97.2%
- **Pengaruh kompresi MP3**: Spectrogram RMSE = 0.02082808129489422, menunjukkan perbedaan kecil pada level fitur akustik.
- **Kedua format menghasilkan beatmap**: MP3 = True, WAV = True. Kedua format tetap menghasilkan beatmap yang dapat digunakan.

### Keterbatasan Pengujian

- Pengujian hanya menggunakan satu pasang lagu.
- Tidak dilakukan uji statistik formal (t-test, dll) untuk jumlah pengulangan yang kecil.
- Perbedaan waktu pemrosesan dapat dipengaruhi oleh beban sistem lain.
- Encoding MP3 dapat bervariasi berdasarkan encoder dan bitrate yang digunakan.

## 11. Kesimpulan Faktual

Berdasarkan hasil pengujian:

1. File MP3 berukuran 1.8928 MB, sedangkan WAV berukuran 19.6472 MB.
2. Rata-rata waktu pipeline MP3 = 1.3570s, WAV = 1.2679s.
3. Spectrogram menunjukkan Pearson correlation = 0.9997767806053162.
4. MP3 menghasilkan 216 note, WAV menghasilkan 216 note.
5. Pada toleransi 50 ms, 210 note cocok berdasarkan waktu.
6. Hasil deterministik antar-run: Ya.
7. Kedua format berhasil menghasilkan beatmap JSON yang valid.

## Peringatan

- ⚠️ File WAV (19.6472 MB) melebihi batas upload backend (10 MB). Pengujian tetap dilanjutkan untuk perbandingan pipeline.

