# BeatBERT Tap-Only Beatmap Generator

Project training-ready untuk pembentukan beatmap otomatis **tap-only** dari audio WAV dan ground-truth MIDI, disusun agar selaras dengan skripsi: **mel-spectrogram -> CNN frontend -> bidirectional Transformer encoder (BERT-like) -> prediksi event/lane -> post-processing -> JSON**.

## Fokus versi ini
- **Tap-only**: tidak ada hold note pada training maupun inference.
- MIDI dipakai sebagai **ground truth training** saja.
- Output inference final adalah **JSON gameplay**, bukan MIDI.
- JSON sudah disiapkan agar mudah dibaca backend / Unity.

## Fitur utama
- Preprocessing dataset WAV + MIDI menjadi fitur log-mel dan label frame-level.
- Parsing MIDI dengan dukungan **tempo map** untuk mengambil waktu note secara akurat.
- Mapping lane yang **configurable** (`explicit`, `modulo`, `range`).
- Model PyTorch nyata: **CNN + positional embedding + Transformer encoder + head event/lane**.
- Training loop lengkap: mixed precision, checkpoint, early stopping, gradient clipping, metrics.
- Augmentasi synthetic yang aman untuk training: **pitch shift** dan **time stretch** di level audio, dengan penyesuaian timestamp label tap-only otomatis saat time stretch.
- Evaluasi frame-level dan event-level.
- Inference per lagu + post-processing agar beatmap lebih playable.
- Export hasil ke JSON yang mudah dipakai backend / Unity.

## Struktur dataset
```bash
project/
  data/
    raw/
      song_001.wav
      song_001.mid
      song_002.wav
      song_002.mid
```

Nama file audio dan MIDI harus memiliki **stem** yang sama.

## Instalasi
```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 1) Edit konfigurasi
Ubah `configs/default.yaml` sesuai dataset Anda, terutama:
- `paths.raw_dir`
- `paths.processed_dir`
- `paths.splits_dir`
- `midi.lane_strategy`
- `midi.explicit_pitch_map` bila pitch MIDI sudah merepresentasikan lane tertentu

Contoh mapping explicit untuk 4 lane:
```yaml
midi:
  lane_strategy: explicit
  explicit_pitch_map:
    36: 0
    38: 1
    42: 2
    46: 3
```

## 2) Build split train/val/test
```bash
python scripts/make_splits.py --config configs/default.yaml
```

## 3) Preprocess semua lagu
```bash
python scripts/preprocess_dataset.py --config configs/default.yaml
```

Output akan masuk ke `data/processed/` berupa file `.npz` per lagu + `metadata.csv`.

### Opsi augmentasi synthetic saat training
Augmentasi diproses di level audio saat preprocessing, lalu file hasil augmentasi otomatis dimasukkan ke split **train** saja.

Contoh konfigurasi:
```yaml
augmentation:
  enabled: true
  include_original: true
  pitch_shift_steps: [-2.0, -1.0, 1.0, 2.0]
  time_stretch_rates: [0.95, 0.98, 1.02, 1.05]
  combine_pitch_and_stretch: false
```

Aturan yang dipakai:
- **Pitch shift**: audio diubah, label waktu note tetap.
- **Time stretch**: audio diubah, dan timestamp label tap-only otomatis diskalakan dengan faktor `1 / rate`.
- Variasi augmentasi hanya dipakai untuk **train split**, sehingga val/test tetap bersih.

## 4) Train model
```bash
python scripts/train.py --config configs/default.yaml
```

Checkpoint terbaik akan disimpan ke `checkpoints/best.pt`.

## 5) Evaluasi
```bash
python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pt
```

## 6) Inference satu lagu
```bash
python scripts/predict.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/best.pt \
  --audio path/to/song.wav \
  --output data/predictions/song.json
```

## Format JSON output
Format utama yang dipakai game: `notes`

```json
{
  "song": "song.wav",
  "bpm": 128.5,
  "offset_ms": 0,
  "num_events": 4,
  "notes": [
    {"time_ms": 540, "lane": 1},
    {"time_ms": 820, "lane": 3},
    {"time_ms": 1120, "lane": 0},
    {"time_ms": 1400, "lane": 2}
  ]
}
```

Field `events` tetap disimpan untuk debugging/post-processing, tetapi untuk integrasi game gunakan `notes`.

## Catatan penting sebelum training
1. **Lane mapping harus benar.** Ini paling krusial. Jika pitch MIDI Anda tidak langsung merepresentasikan lane, ubah konfigurasi mapping dulu.
2. Pipeline ini **tap-only**, jadi semua note training diperlakukan sebagai tap.
3. Jika augmentasi diaktifkan, jalankan urutan: **make_splits -> preprocess_dataset -> train**. Split harus dibuat dulu agar augmentasi hanya masuk train set.
4. Untuk skripsi, saya sarankan mulai dari subset kecil dulu (5-10 lagu) untuk validasi pipeline end-to-end.
5. Model ini nyata dan trainable, tetapi kualitas akhir tetap tergantung kebersihan sinkronisasi dataset WAV-MIDI.

## Rekomendasi integrasi game
- Training: WAV + MIDI ground truth
- Inference: WAV -> JSON `notes`
- Game: baca JSON `notes` langsung, tanpa konversi kembali ke MIDI

## Referensi desain
- Skripsi Anda: fokus pada beat utama/onset, lane, post-processing, JSON output, dan integrasi backend-Unity.
- BeatLearning: referensi konseptual beatmap generation berbasis transformer untuk rhythm-game sequence.
