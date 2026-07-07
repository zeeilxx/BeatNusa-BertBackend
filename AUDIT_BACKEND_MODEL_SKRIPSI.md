# AUDIT TEKNIS BACKEND DAN MODEL BEATNUSA (BEATMAPBERT)
*Dokumentasi Faktual untuk Penulisan Bab III dan Bab IV Skripsi*

---

## 1. Ringkasan Eksekutif

Audit teknis ini dilakukan secara menyeluruh terhadap repositori proyek backend dan model AI **BeatmapBERT** yang terintegrasi dengan game musik **BeatNusa**. Backend dibangun menggunakan framework **FastAPI**, basis data **TiDB Cloud (MySQL-compatible)**, dan pipeline deep learning berbasis **PyTorch** (CNN + Transformer Encoder).

Hasil utama dari audit ini meliputi verifikasi arsitektur model, parameter preprocessing audio, struktur database riil pada TiDB Cloud, spesifikasi endpoint FastAPI, detail parameter post-processing aktif, dan evaluasi checkpoint terbaik (`best.pt`) pada Epoch 30. Beberapa klaim rancangan awal skripsi didapati berbeda dengan kondisi riil kode produksi, yang mana direkapitulasi secara rinci pada bagian akhir dokumen ini demi menjaga kesesuaian penulisan naskah skripsi Bab III dan Bab IV.

---

## 2. Struktur Proyek

Struktur folder aktual dari proyek backend adalah sebagai aslinya sebagai berikut:

```text
beatmap_bert_project_real - Copy/
├── app/                             # Kode utama aplikasi backend FastAPI
│   ├── config.py                    # Konfigurasi aplikasi & settings (.env)
│   ├── database.py                  # Inisialisasi engine database & session maker SQLAlchemy
│   ├── main.py                      # File entri utama aplikasi (App Factory & Lifespan)
│   ├── models/                      # Definisi ORM Model SQLAlchemy
│   │   ├── song.py
│   │   ├── beatmap.py
│   │   └── game_result.py
│   ├── routers/                     # Router API FastAPI
│   │   ├── songs.py
│   │   ├── beatmaps.py
│   │   └── game_results.py
│   ├── schemas/                     # Skema validasi Pydantic
│   └── services/                    # Logika bisnis inti backend
│       ├── audio_service.py
│       ├── beatmap_service.py
│       └── ai_service.py
├── checkpoints/                     # Direktori penyimpanan bobot model latih
│   ├── best.pt                      # Bobot model dengan F1-score terbaik
│   └── last.pt                      # Bobot model dari epoch terakhir
├── configs/                         # File konfigurasi YAML untuk training/inference
│   ├── default.yaml
│   └── local.yaml                   # Konfigurasi aktif (RTX 3050 Laptop / Lokal)
├── data/                            # Folder data
│   ├── predictsongs/                # File audio untuk inferensi (misal: malumalu.wav)
│   ├── processed/                   # Dataset dalam format .npz & metadata.csv
│   └── splits/                      # File CSV pembagian dataset (train, val, test)
├── scripts/                         # Script utility dan diagnostik luar
│   ├── evaluate.py                  # Script evaluasi model terhadap test set
│   ├── train.py                     # Script pemicu training model
│   ├── read_checkpoint_metrics.py   # Script pembaca metrik checkpoint
│   └── generate_thesis_artifacts.py # Script pembuat artefak diagnostik Bab IV
├── src/                             # Modul python BeatBERT untuk deep learning
│   └── beatbert/
│       ├── configs/
│       ├── data/
│       ├── inference/
│       │   ├── predictor.py         # Logika inferensi chunked
│       │   └── postprocess.py       # Logika snapping, gap filter, & density
│       ├── models/
│       │   ├── cnn_frontend.py      # Layer ekstraksi fitur spasio-temporal CNN
│       │   ├── transformer.py       # Positional embedding & Transformer Encoder
│       │   └── beatmap_model.py     # Class model gabungan
│       └── utils/
│           ├── audio.py             # Preprocessing & ekstraksi log-mel/onset/beat
│           └── midi.py              # Parsing file MIDI & pembentukan label
└── storage/                         # Direktori lokal penyimpanan file audio yang diupload
    └── audio/
```

---

## 3. Arsitektur Backend

Aplikasi backend berjalan di atas **FastAPI** dengan siklus startup sebagai berikut:
1. Mengaktifkan loop kebijakan event Windows Selector jika didektesi berjalan di Windows ([app/main.py:L10-12](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/app/main.py#L10-L12)).
2. Menyiapkan database secara asinkron lewat SQLAlchemy dengan driver `aiomysql` ([app/database.py:L17-24](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/app/database.py#L17-L24)).
3. Menggunakan fungsi `lifespan` ([app/main.py:L27-60](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/app/main.py#L27-L60)) untuk:
   * Menjalankan migrasi pembuatan tabel database secara otomatis ([app/database.py:L53-56](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/app/database.py#L53-L56)).
   * Memuat model AI BeatmapBERT ke memori GPU/CPU ([app/services/ai_service.py:L41-62](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/app/services/ai_service.py#L41-L62)).
   * Menjamin direktori penyimpanan audio lokal telah siap ([app/config.py:L60-61](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/app/config.py#L60-L61)).

---

## 4. Daftar Endpoint Aktual

Berikut adalah daftar endpoint aktual yang terdaftar pada aplikasi FastAPI:

| No | HTTP Method | Path Endpoint | Fungsi Handler | Lokasi Definisi File | Cara Pemanggilan oleh Unity / Client | Sync / Async | Database / File Storage yang Terlibat |
|---|---|---|---|---|---|---|---|
| 1 | `GET` | `/` | `health_check` | [app/main.py:L95](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/app/main.py#L95) | Pengecekan server hidup / inisialisasi status AI. | Sync | Tidak ada |
| 2 | `POST` | `/api/songs/upload` | `upload_song` | [app/routers/songs.py:L30](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/app/routers/songs.py#L30) | Upload file audio. Mengembalikan status 202 Accepted secara instan, sedangkan pemrosesan AI dijalankan di background thread. | Async (Inference di background thread) | Tabel `songs`, direktori `storage/audio` |
| 3 | `GET` | `/api/songs` | `list_songs` | [app/routers/songs.py:L77](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/app/routers/songs.py#L77) | Diambil oleh Unity untuk menampilkan katalog lagu yang siap dimainkan (status = `'done'`). | Async | Tabel `songs` |
| 4 | `GET` | `/api/songs/{song_code}/status` | `get_song_status` | [app/routers/songs.py:L123](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/app/routers/songs.py#L123) | Polling oleh Unity/UI setelah upload untuk mendeteksi kapan proses AI selesai. | Async | Tabel `songs` |
| 5 | `GET` | `/api/songs/{song_code}/audio{ext}` | `get_song_audio` | [app/routers/songs.py:L148](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/app/routers/songs.py#L148) | Diambil oleh Unity untuk memuat/streaming file audio permainan. | Async | Tabel `songs`, file audio di `storage/audio` |
| 6 | `GET` | `/api/beatmaps/{song_code}` | `get_beatmap` | [app/routers/beatmaps.py:L30](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/app/routers/beatmaps.py#L30) | Diambil oleh Unity untuk memuat catatan note (beatmap) yang digenerate AI. | Async | Tabel `beatmaps`, `songs` |
| 7 | `POST` | `/api/beatmaps/{song_code}/regenerate` | `regenerate_beatmap_endpoint` | [app/routers/beatmaps.py:L93](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/app/routers/beatmaps.py#L93) | Memaksa pembuatan ulang beatmap. Eksekusi memblokir response HTTP hingga proses AI selesai. | Async (Memblokir executor pool) | Tabel `beatmaps`, `songs` |
| 8 | `POST` | `/api/game-results` | `submit_game_result` | [app/routers/game_results.py:L29](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/app/routers/game_results.py#L29) | Dikirim oleh Unity setelah lagu selesai dimainkan untuk mencatat skor/akurasi pemain. | Async | Tabel `game_results`, `songs`, `beatmaps` |
| 9 | `GET` | `/api/game-results/{song_code}` | `get_game_results` | [app/routers/game_results.py:L106](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/app/routers/game_results.py#L106) | Mengambil riwayat skor permainan untuk lagu tertentu. | Async | Tabel `game_results`, `songs` |
| 10 | `GET` | `/api/game-results/{song_code}/leaderboard` | `get_leaderboard` | [app/routers/game_results.py:L156](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/app/routers/game_results.py#L156) | Diambil oleh Unity untuk menampilkan 10 peringkat skor tertinggi pada lagu tersebut. | Async | Tabel `game_results`, `songs` |

> [!WARNING]
> Endpoint `GET /api/songs/{song_code}` **tidak aktif/dikomentari** pada kode produksi ([app/routers/songs.py:L103-116](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/app/routers/songs.py#L103-L116)) dan dialihkan fungsinya langsung ke endpoint detail status atau beatmap.

---

## 5. Validasi Upload Audio

Prosedur pengunggahan audio divalidasi ketat melalui module [app/services/audio_service.py](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/app/services/audio_service.py):
* **Format Diterima:** `.mp3`, `.wav`, `.ogg` ([app/services/audio_service.py:L17](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/app/services/audio_service.py#L17)).
* **Ukuran Maksimal:** **50 MB** riil, karena didefinisikan di `.env` lewat parameter `MAX_UPLOAD_SIZE_MB=50`, yang menimpa nilai default 10 MB pada `config.py` ([app/config.py:L31](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/app/config.py#L31)).
* **Durasi Maksimal:** **600 detik** (10 menit) ([app/config.py:L32](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/app/config.py#L32)).
* **File Kosong:** Diperiksa melalui parameter `file.filename` ([app/services/audio_service.py:L31](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/app/services/audio_service.py#L31)).
* **Kode Lagu (`song_code`):** Dibuat otomatis dalam format `SONG-{uuid.uuid4().hex[:8].upper()}` (contoh: `SONG-B31F7B4A`) ([app/services/beatmap_service.py:L62](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/app/services/beatmap_service.py#L62)).
* **Transcoding:** File non-WAV (seperti MP3 dan OGG) akan otomatis ditranskode oleh backend menjadi format WAV mono PCM_16 menggunakan bantuan pustaka `librosa` dan `soundfile` demi menjamin stabilitas pembacaan audio pada FMOD Unity WebGL ([app/services/audio_service.py:L79-98](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/app/services/audio_service.py#L79-L98)).
* **Status Pemrosesan Lagu:**
  * `uploaded`: Lagu baru diunggah dan disimpan ke penyimpanan server.
  * `processing`: Lagu sedang dibaca fitur audionya dan diprediksi oleh model AI.
  * `done`: Beatmap selesai diolah dan disimpan ke basis data, siap dimainkan.
  * `failed`: Terjadi kegagalan pemrosesan fitur atau kegagalan model AI.

---

## 6. Struktur Database

Struktur database riil yang digunakan pada **TiDB Cloud** berhasil diverifikasi melalui perintah `SHOW CREATE TABLE`:

### A. Tabel `songs`
Tabel ini digunakan untuk menyimpan metadata audio yang telah berhasil diunggah.

| Nama Kolom | Tipe Data Rujukan (SQL) | Nullable | Default | Keterangan / Constraints |
|---|---|---|---|---|
| `id` | `bigint unsigned` | NO | *None* | `PRIMARY KEY`, `AUTO_INCREMENT` |
| `song_code` | `varchar(100)` | NO | *None* | `UNIQUE KEY`, `INDEX` |
| `title` | `varchar(255)` | NO | *None* | `INDEX` (`idx_songs_title`) |
| `artist` | `varchar(255)` | YES | NULL | Nama artis/musisi |
| `genre` | `varchar(100)` | YES | NULL | Genre lagu |
| `original_filename` | `varchar(255)` | NO | *None* | Nama file asli saat diupload |
| `stored_filename` | `varchar(255)` | NO | *None* | Nama file unik yang disimpan di server |
| `file_path` | `varchar(500)` | NO | *None* | Path lokasi fisik file audio di server |
| `file_format` | `enum('mp3','wav')` | NO | *None* | Jenis format file audio asli |
| `duration_seconds` | `decimal(10,2)` | YES | NULL | Durasi total audio dalam detik |
| `bpm` | `decimal(10,2)` | YES | NULL | Nilai tempo (BPM) hasil analisis audio |
| `cover_image_path` | `varchar(500)` | YES | NULL | Path gambar sampul lagu |
| `upload_source` | `enum('user_upload','seeded','admin')` | NO | `'user_upload'` | Asal pengunggahan |
| `process_status` | `enum('uploaded','processing','done','failed')` | NO | `'uploaded'` | Status pemrosesan AI, `INDEX` (`idx_songs_status`) |
| `is_active` | `tinyint(1)` | NO | `'1'` | Flag aktif (soft delete) |
| `created_at` | `timestamp` | NO | `CURRENT_TIMESTAMP` | Tanggal data dibuat |
| `updated_at` | `timestamp` | NO | `CURRENT_TIMESTAMP` | Diupdate otomatis (`ON UPDATE CURRENT_TIMESTAMP`) |

### B. Tabel `beatmaps`
Tabel ini menampung beatmap hasil bentukan model AI.

| Nama Kolom | Tipe Data Rujukan (SQL) | Nullable | Default | Keterangan / Constraints |
|---|---|---|---|---|
| `id` | `bigint unsigned` | NO | *None* | `PRIMARY KEY`, `AUTO_INCREMENT` |
| `song_id` | `bigint unsigned` | NO | *None* | `FOREIGN KEY` -> `songs(id)` ON DELETE CASCADE, `INDEX` (`idx_beatmaps_song_id`) |
| `model_name` | `varchar(100)` | NO | *None* | Nama model AI (Default: `'BeatmapBERT'`) |
| `model_version` | `varchar(100)` | YES | NULL | Versi model (Default: `'1.0'`) |
| `difficulty_name` | `varchar(100)` | NO | `'normal'` | Tingkat kesulitan beatmap |
| `lane_count` | `int` | NO | `'4'` | Jumlah lajur tombol |
| `offset_ms` | `int` | NO | `'0'` | Offset waktu mulai permainan (ms) |
| `beatmap_json` | `longtext` | NO | *None* | Payload beatmap terenkode JSON string |
| `note_count` | `int` | YES | NULL | Jumlah total note yang terbentuk |
| `generation_status` | `enum('generated','validated','failed')` | NO | `'generated'` | Status pembuatan beatmap |
| `validation_notes` | `text` | YES | NULL | Catatan validasi beatmap |
| `generated_at` | `timestamp` | NO | `CURRENT_TIMESTAMP` | Waktu generate |
| `updated_at` | `timestamp` | NO | `CURRENT_TIMESTAMP` | Diupdate otomatis (`ON UPDATE CURRENT_TIMESTAMP`) |

* *Constraint Tambahan:* `UNIQUE KEY uq_song_difficulty` (`song_id`, `difficulty_name`) menjamin satu lagu hanya memiliki satu beatmap aktif per tingkat kesulitan.

### C. Tabel `game_results`
Tabel ini digunakan untuk mencatat riwayat permainan dari sisi client (Unity).

| Nama Kolom | Tipe Data Rujukan (SQL) | Nullable | Default | Keterangan / Constraints |
|---|---|---|---|---|
| `id` | `bigint unsigned` | NO | *None* | `PRIMARY KEY`, `AUTO_INCREMENT` |
| `song_id` | `bigint unsigned` | NO | *None* | `FOREIGN KEY` -> `songs(id)` ON DELETE CASCADE, `INDEX` |
| `beatmap_id` | `bigint unsigned` | NO | *None* | `FOREIGN KEY` -> `beatmaps(id)` ON DELETE CASCADE, `INDEX` |
| `player_name` | `varchar(100)` | YES | NULL | Nama pemain |
| `score` | `int` | NO | `'0'` | Nilai skor akhir |
| `accuracy` | `decimal(5,2)` | NO | `'0.00'` | Persentase akurasi ketukan (%) |
| `max_combo` | `int` | NO | `'0'` | Combo ketukan berturut-turut tertinggi |
| `hit_count` | `int` | NO | `'0'` | Jumlah total ketukan sukses |
| `miss_count` | `int` | NO | `'0'` | Jumlah ketukan meleset (*Miss*) |
| `good_count` | `int` | NO | `'0'` | Jumlah ketukan berpredikat *Good* |
| `perfect_count` | `int` | NO | `'0'` | Jumlah ketukan berpredikat *Perfect* |
| `bad_count` | `int` | NO | `'0'` | Jumlah ketukan berpredikat *Bad* |
| `mean_offset_ms` | `decimal(10,2)` | YES | NULL | Rata-rata selisih ketukan pemain (ms) |
| `played_at` | `timestamp` | NO | `CURRENT_TIMESTAMP` | Waktu permainan dimainkan |

---

## 7. Query Database Aktual

Berikut adalah operasi kueri database aktual yang terdeteksi pada kode program:
* **Menyimpan Metadata Lagu:** ([app/services/beatmap_service.py:L65-82](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/app/services/beatmap_service.py#L65-L82))
  `db.add(song)` diikuti `await db.commit()`.
* **Memperbarui Status Proses:** ([app/services/beatmap_service.py:L100-101](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/app/services/beatmap_service.py#L100-L101))
  `song.process_status = "processing"` lalu `await db.commit()`. Jika sukses diubah ke `"done"`, jika gagal diubah ke `"failed"`.
* **Menyimpan Beatmap Baru:** ([app/services/beatmap_service.py:L114-128](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/app/services/beatmap_service.py#L114-L128))
  `db.add(beatmap)` diikuti `await db.commit()`.
* **Mengambil Lagu Siap Dimainkan:** ([app/routers/songs.py:L88-95](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/app/routers/songs.py#L88-L95))
  `select(Song).where(Song.is_active == True).where(Song.process_status == "done").order_by(Song.created_at.desc())`.
* **Mengambil Beatmap Berdasarkan `song_code`:** ([app/services/beatmap_service.py:L137-148](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/app/services/beatmap_service.py#L137-L148))
  `select(Beatmap).join(Song, Beatmap.song_id == Song.id).where(Song.song_code == song_code).where(Song.process_status == "done").where(Beatmap.generation_status == "generated").order_by(Beatmap.generated_at.desc()).limit(1)`.
* **Menyimpan Hasil Permainan:** ([app/routers/game_results.py:L67-82](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/app/routers/game_results.py#L67-L82))
  `db.add(game_result)` diikuti `await db.commit()`.
* **Mengambil Leaderboard:** ([app/routers/game_results.py:L178-185](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/app/routers/game_results.py#L178-L185))
  `select(GameResult).where(GameResult.song_id == song.id).order_by(desc(GameResult.score)).limit(limit)`.

---

## 8. Pipeline Preprocessing Audio

Proses preprocessing audio dari file fisik hingga menjadi input tensor bagi model model deep learning adalah sebagai berikut ([src/beatbert/utils/audio.py:L69-86](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/src/beatbert/utils/audio.py#L69-L86)):

```mermaid
graph TD
    A[File Audio MP3/WAV/OGG] -->|Transcode & Load| B[y, sr=22050 Mono Waveform]
    B -->|Librosa melspectrogram| C[Power Spectrogram]
    C -->|librosa.power_to_db| D[Log-Mel Spectrogram in dB]
    D -->|Z-Score Standardisation| E[Normalized Log-Mel Spectrogram]
    E -->|Tensor conversion| F[Input Tensor [1, 128, T]]
```

### Parameter Preprocessing Konfigurasi Aktif (`local.yaml`)
* **Pustaka Utama:** `librosa` ([src/beatbert/utils/audio.py:L7](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/src/beatbert/utils/audio.py#L7))
* **Sample Rate:** `22050` Hz
* **Channel:** Mono (`mono: true`)
* **Window FFT Size (`n_fft`):** `2048`
* **Hop Length:** `256` (artinya pergeseran jendela sejauh 256 sampel)
* **Window Length (`win_length`):** `2048`
* **Mel Bands (`n_mels`):** `128`
* **Rentang Frekuensi Mel:** `fmin: 30` Hz s.d `fmax: 11025` Hz
* **Puncak Desibel Spektrogram (`top_db`):** `80` dB
* **Normalisasi Spektrogram:** Z-score normalization ([src/beatbert/utils/audio.py:L46](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/src/beatbert/utils/audio.py#L46)):
  $$\text{mel\_db} = \frac{\text{mel\_db} - \text{mean}}{\text{std} + 1e-8}$$
* **Durasi Satu Frame:**
  $$\text{Durasi Frame} = \frac{\text{hop\_length}}{\text{sample\_rate}} = \frac{256}{22050} \approx 11.61\text{ ms}$$
* **Durasi Satu Jendela Konteks (Context Window):** `256` frame $\approx 2972.15\text{ ms}$ (atau $\approx 2.97$ detik).

---

## 9. Dataset dan Split Aktual

* **Sumber Dataset:** Maestro Piano Dataset (file audio `.mp3`/`.wav` dipasangkan dengan file anotasi nada `.midi` asli dari keyboard MIDI).
* **Jumlah Dataset Riil:** Terdiri dari **1000 lagu** total ([data/processed/metadata.csv](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/data/processed/metadata.csv)).
* **Pembagian Dataset:**
  * **Train Split:** **699** lagu (file [train.csv](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/data/splits/train.csv))
  * **Validation Split:** **150** lagu (file [val.csv](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/data/splits/val.csv))
  * **Test Split:** **151** lagu (file [test.csv](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/data/splits/test.csv))
* **Metode Split:** Menggunakan `train_test_split` acak berdasarkan **`song_id`** (pemisah lagu penuh) dengan seed `42` ([src/beatbert/data/splits.py:L18-20](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/src/beatbert/data/splits.py#L18-L20)). Metode pembagian per lagu penuh ini **mencegah kebocoran data (data leakage)** secara absolut (dibandingkan pembagian berbasis potongan frame lagu yang sama).
* **Negative Sample Ratio:** `0.30` ([configs/local.yaml:L35](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/configs/local.yaml#L35)). Untuk menyeimbangkan porsi data kosong pada training, model hanya memuat potongan frame tanpa note sebanyak maksimal 30% dari total jumlah frame aktif yang memiliki note ([src/beatbert/data/dataset.py:L68-72](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/src/beatbert/data/dataset.py#L68-L72)).
* **Strategi Mapping Pitch:** `range` ([configs/local.yaml:L40](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/configs/local.yaml#L40)). Rentang nada piano standar [21, 108] dipetakan merata ke 4 lane (0, 1, 2, 3) ([src/beatbert/utils/midi.py:L31-36](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/src/beatbert/utils/midi.py#L31-L36)).
* **Tipe Note:** Sistem **hanya memetakan Note On (tap note)**. Model tidak melatih struktur hold note (panjang) karena durasi note diabaikan pada penyusunan label dataset ([src/beatbert/utils/midi.py:L102-105](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/src/beatbert/utils/midi.py#L102-L105)).

---

## 10. Arsitektur Model

Arsitektur model AI BeatmapBERT tersusun atas 3 bagian utama:
1. **CNN Frontend:** Menerima input tensor spektrogram, menyempitkan dimensi spasial mel, dan mengekstrak representasi temporal fitur ([src/beatbert/models/cnn_frontend.py](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/src/beatbert/models/cnn_frontend.py)).
2. **Positional Embedding:** Menambahkan penyandian posisi urutan ke tensor CNN ([src/beatbert/models/transformer.py:L7-15](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/src/beatbert/models/transformer.py#L7-L15)).
3. **Transformer Encoder:** Menganalisis korelasi konteks antartingkat ketukan di setiap frame waktu ([src/beatbert/models/transformer.py:L18-36](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/src/beatbert/models/transformer.py#L18-L36)).

### Alur Perubahan Bentuk Tensor (Tensor Shapes Progression)
Berikut adalah alur transformasi tensor dalam satu forward pass model (dengan batch size = $B$ dan sequence length = $T$):

$$\begin{aligned}
\text{Normalized Log-Mel Input} \quad & [B, 128, T] \\
\downarrow \quad & \text{(Permute / Channel unsqueeze)} \\
& [B, 1, 128, T] \\
\downarrow \quad & \text{(Conv2d Channels } c_1=32 \text{ + BatchNorm + GELU)} \\
& [B, 32, 128, T] \\
\downarrow \quad & \text{(Conv2d Channels } c_2=64 \text{ + BatchNorm + GELU)} \\
& [B, 64, 128, T] \\
\downarrow \quad & \text{(MaxPool2d Jendela } (2, 1) \text{ - mel halved)} \\
& [B, 64, 64, T] \\
\downarrow \quad & \text{(Conv2d Channels } c_3=128 \text{ + BatchNorm + GELU)} \\
& [B, 128, 64, T] \\
\downarrow \quad & \text{(MaxPool2d Jendela } (2, 1) \text{ - mel halved)} \\
& [B, 128, 32, T] \\
\downarrow \quad & \text{(Permute to } [B, T, C, M'] \text{)} \\
& [B, T, 128, 32] \\
\downarrow \quad & \text{(Flatten channel \& mel space)} \\
& [B, T, 4096] \\
\downarrow \quad & \text{(Linear Projection to } d_{model} \text{)} \\
& [B, T, 128] \\
\downarrow \quad & \text{(Add Positional Embedding)} \\
& [B, T, 128] \\
\downarrow \quad & \text{(BeatTransformerEncoder } 2 \text{ Layers, } 4 \text{ Heads, } d_{ff}=256\text{)} \\
& [B, T, 128] \\
\downarrow \quad & \text{(LayerNorm)} \\
& [B, T, 128] \\
\text{Output Heads:} \quad & \begin{cases}
\text{event\_logits} & \rightarrow [B, T] \quad \text{(lewat Linear[128, 1] + Squeeze)} \\
\text{lane\_logits} & \rightarrow [B, T, 4] \quad \text{(lewat Linear[128, 4])}
\end{cases}
\end{aligned}$$

---

## 11. Pemeriksaan Checkpoint

Bobot checkpoint aktif diparkir pada file [checkpoints/best.pt](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/checkpoints/best.pt). Hasil pembedahan data biner checkpoint menunjukkan parameter sebagai berikut:

* **Struktur Kunci Checkpoint:**
  `dict_keys(['epoch', 'model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'config', 'val_metrics', 'val_loss'])`
* **Epoch Terakhir Terlatih:** **30**
* **Metrik Validasi Terbaik:**
  * **Precision:** `0.565462`
  * **Recall:** `0.856888`
  * **F1-score:** `0.680947`
  * **Lane Accuracy:** `0.867493`
  * **Validation Loss:** `0.256781`
* **Monitor Metric:** Dikonfigurasi berbasis **F1-score** (`f1`) untuk memprioritaskan penanganan ketimpangan data kelas.
* **Kecocokan Arsitektur:** Struktur state dict model sepenuhnya cocok dengan konfigurasi aktif di [configs/local.yaml](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/configs/local.yaml):
  * `d_model = 128`
  * `num_heads = 4`
  * `num_layers = 2`
  * `ff_dim = 256` (`ff_mult = 2`)
  * `num_lanes = 4`

---

## 12. Konfigurasi Training Final

Berdasarkan checkpoint `best.pt`, parameter training final yang digunakan adalah:
* **Seed:** `42`
* **Epoch Maksimal:** `40` (Training dihentikan/terhenti pada Epoch `30`)
* **Batch Size:** `16`
* **Learning Rate:** `0.0003`
* **Weight Decay:** `0.02`
* **Warmup Epochs:** `5`
* **Scheduler:** Cosine Annealing dengan Linear Warmup (`LambdaLR`)
* **AMP (Automatic Mixed Precision):** `true` (Fp16 diaktifkan untuk melatih model pada laptop VRAM kecil 4GB)
* **Gradient Clipping:** `1.0`
* **Event Loss Weight:** `1.0`
* **Lane Loss Weight:** `0.5`
* **Label Smoothing:** `0.05`
* **Loss Function Event:** Focal Loss (dengan parameter `focal_alpha: 0.65` dan `focal_gamma: 2.0`)
* **Loss Function Lane:** Standard Cross Entropy (mengabaikan loss kelas `-100` pada silent frame)

---

## 13. Hasil Training yang Dapat Diverifikasi

Proses training **tidak menulis log ke CSV / TensorBoard**. Data validasi historis per epoch yang terverifikasi hanya berasal dari checkpoint status akhir yang disimpan pada epoch 30:

| Epoch | Train Loss | Validation Loss | Precision | Recall | F1-score | Lane Accuracy | Keterangan |
|---|---|---|---|---|---|---|---|
| 30 | *Tidak tersimpan* | 0.256781 | 0.565462 | 0.856888 | 0.680947 | 0.867493 | Checkpoint Terbaik (best.pt) |

---

## 14. Evaluasi Validation dan Test Set

* **Metrik Validasi (Validation Set):** Dihitung pada validation set menggunakan ambang batas evaluasi `event_threshold = 0.42` ([configs/local.yaml:L85](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/configs/local.yaml#L85)). Hasil validasinya adalah F1-score `0.6809` dan Lane Accuracy `0.8675`.
* **Metrik Evaluasi Test Set:** Script evaluasi terpisah [scripts/evaluate.py](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/scripts/evaluate.py) dipersiapkan untuk menguji model pada test split (151 lagu). Evaluasi ini mengadopsi:
  * **Timing Tolerance:** Nilai toleransi jarak waktu `tolerance_ms` didefinisikan lewat konfigurasi.
  * **Lane Accuracy Test:** Dihitung sebagai `lane_event_acc` (jumlah note dengan lane benar dibagi total note terprediksi benar / TP).
  * **Macro/Micro:** Hasil dihitung bertahap per lagu kemudian dirata-rata (macro-averaging secara tidak langsung pada tingkat dataset).

---

## 15. Pipeline Inference

Proses inferensi end-to-end yang berjalan di backend untuk menghasilkan beatmap dari unggahan lagu adalah sebagai berikut:

```text
1. Unggah Audio MP3/WAV/OGG -> Disimpan di storage/audio
2. Transcoding otomatis ke WAV mono 22050Hz (jika format awal non-WAV)
3. Ekstraksi Fitur Audio -> log-mel spectrogram [128, T] & rhythm guides (onset & beat times)
4. Chunking Spectrogram -> Spektrogram dipotong per 512 frame dengan overlap 128 frame (Stride = 384 frame)
5. Padding Akhir -> Potongan terakhir yang kurang dari 512 frame dipad dengan nol
6. Forward Pass Model -> forward(mel, attention_mask) menghasilkan event_logits & lane_logits
7. Logits Activation -> event_prob = sigmoid(event_logits), lane_prob = softmax(lane_logits, dim=-1)
8. Output Reconstruction -> Rata-rata probabilitas antar chunk yang bertumpukan dihitung ulang
9. Thresholding -> Frame dengan event_prob >= 0.60 diambil sebagai kandidat note
10. Post-processing -> Snapping ke onset/beat terdekat, filter gap global/lane, & batas densitas
11. Beatmap Formatting -> Format note menjadi JSON objek
12. Database Storage -> JSON string disimpan ke field 'beatmap_json' di tabel 'beatmaps'
```

---

## 16. Post-Processing

Penyaringan dan pembenahan note mentah luaran model diproses secara urut di [src/beatbert/inference/postprocess.py](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/src/beatbert/inference/postprocess.py):
1. **Event Thresholding:** Note disaring dengan threshold `event_threshold: 0.60` (sangat selektif untuk gameplay).
2. **Onset Snapping:** Note digeser waktunya ke waktu onset fisik lagu terdekat dalam batas toleransi `onset_snap_tolerance_ms: 30` ms.
3. **Beat Snapping:** Note kemudian digeser ke ketukan beat lagu terdekat dalam batas toleransi `beat_snap_tolerance_ms: 40` ms.
4. **Global Gap Filter:** Jika jarak note ke note sebelumnya kurang dari `min_gap_ms: 200` ms, note baru akan dibuang.
5. **Same-Lane Gap Filter:** Jika jarak note ke note sebelumnya pada lajur lane yang sama kurang dari `same_lane_min_gap_ms: 250` ms, note baru akan dibuang.
6. **Density Filter:** Menjamin jumlah note dalam rentang 1 detik (1000 ms) tidak melebihi `max_density_notes_per_second: 3` (membatasi tingkat kesulitan beatmap agar tidak terlalu padat).

---

## 17. Struktur Beatmap JSON

Struktur data JSON beatmap disimpan sebagai string teks pada field `beatmap_json` di tabel `beatmaps` dengan format:
```json
{
  "lane_count": 4,
  "offset_ms": 0,
  "notes": [
    {
      "time_ms": 2972,
      "lane": 1,
      "type": "tap",
      "length_ms": 0
    },
    {
      "time_ms": 3530,
      "lane": 3,
      "type": "tap",
      "length_ms": 0
    }
  ]
}
```
* **Rentang Lane:** Integer `[0, 3]`.
* **Jenis Note:** Hanya `"tap"` (ketukan biasa). Proyek ini tidak menggunakan slide/hold note.
* **Offset Waktu (`offset_ms`):** Bernilai konstan `0` karena tidak ada kalkulasi pergeseran waktu lagu di awal backend.

---

## 18. Integrasi Backend-Model-Database

```
[Unity WebGL Client]
      │
      ├── (1) POST /api/songs/upload ──────────────────> [FastAPI Server]
      │                                                        │
      │                                                Transcode ke WAV
      │                                                Simpan data di DB (status: 'uploaded')
      │                                                        │
      │                                             (Tugas Latar Belakang)
      │                                                        ▼
      ├── (2) GET /api/songs/{code}/status <── (Polling)── [Process AI Background]
      │                                             1. Ekstrak Spektrogram
      │                                             2. Inference Model PyTorch
      │                                             3. Post-Processing & Snapping
      │                                             4. Simpan Beatmap JSON ke DB
      │                                             5. Update DB (status: 'done')
      │                                                        ▼
      ├── (3) GET /api/beatmaps/{code} ───────────────> Ambil Beatmap JSON
      │
      └── (4) POST /api/game-results ─────────────────> Catat Hasil Game & Skor
```

---

## 19. Ketidaksesuaian dengan Dokumen Skripsi

Berikut rekapitulasi perbedaan klaim teoritis / draf skripsi dengan kondisi riil kode program:

1. **Format Upload Audio**
   * *Klaim Skripsi:* Hanya mendukung unggahan file MP3/WAV.
   * *Kondisi Riil:* Program juga mendukung file `.ogg` secara eksplisit ([app/services/audio_service.py:L17](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/app/services/audio_service.py#L17)). Namun, file format non-WAV akan langsung ditranskode ke WAV.
   * *Rekomendasi Naskah:* *"Sistem menerima input audio berformat MP3, WAV, dan OGG, di mana format non-WAV dikonversi otomatis oleh backend menjadi WAV mono 22050Hz."*

2. **Batas Ukuran Upload**
   * *Klaim Skripsi:* Batas ukuran file audio adalah 10 MB.
   * *Kondisi Riil:* Nilai default settings memang 10 MB, tetapi file konfigurasi aktif `.env` mengubah batas maksimal menjadi **50 MB** (`MAX_UPLOAD_SIZE_MB=50`).
   * *Rekomendasi Naskah:* *"Batas maksimum file audio yang diperbolehkan diunggah diatur sebesar 50 MB guna mengakomodasi file musik berdurasi panjang."*

3. **Jumlah Epoch Pelatihan**
   * *Klaim Skripsi:* Training diselesaikan penuh hingga 40 epoch.
   * *Kondisi Riil:* Checkpoint terbaik dan terakhir yang tersimpan berada pada **Epoch 30**. Hal ini menunjukkan pelatihan terhenti (early stop / interupsi manual) di epoch 30.
   * *Rekomendasi Naskah:* *"Pelatihan model dieksekusi dengan rencana awal 40 epoch, namun konvergensi optimal tercapai pada epoch 30, yang kemudian disimpan sebagai model final."*

4. **Dimensi Model (`d_model`) dan Hyperparameter Transformer**
   * *Klaim Skripsi (Umum):* Menggunakan d_model = 256, 4 layer Transformer, dan 8 attention heads.
   * *Kondisi Riil:* Diatur pada `local.yaml` untuk menghemat VRAM: `d_model = 128`, `num_layers = 2`, `num_heads = 4`, dan `ff_dim = 256` ([configs/local.yaml:L51-57](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/configs/local.yaml#L51-L57)). Bobot checkpoint terverifikasi memiliki shape yang cocok dengan parameter VRAM hemat ini.
   * *Rekomendasi Naskah:* *"Untuk efisiensi komputasi pada GPU dengan VRAM terbatas (4GB), arsitektur disesuaikan dengan dimensi d_model sebesar 128, 2 layer Transformer Encoder, dan 4 attention heads."*

5. **Ambang Batas (Threshold) Prediksi**
   * *Klaim Skripsi:* Evaluasi dan post-processing memiliki threshold yang sama.
   * *Kondisi Riil:* Tahap evaluasi model menggunakan threshold rendah `0.42` ([configs/local.yaml:L85](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/configs/local.yaml#L85)), sedangkan post-processing gameplay menggunakan threshold selektif `0.60` ([configs/local.yaml:L88](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/configs/local.yaml#L88)).
   * *Rekomendasi Naskah:* *"Model dievaluasi menggunakan ambang batas sensitivitas event sebesar 0.42 untuk menangkap sebanyak mungkin ketukan potensial, sedangkan pembentukan beatmap permainan sesungguhnya menggunakan threshold selektif sebesar 0.60 untuk menjamin kualitas gameplay."*

6. **Parameter Pembatas Ketukan (Post-processing)**
   * *Klaim Skripsi:* Gap minimum global adalah 95 ms, gap per lane 130 ms, densitas 7 note/detik.
   * *Kondisi Riil:* Terbaca pada `local.yaml` (Easy mode): gap global `200` ms, gap per lane `250` ms, dan densitas maksimum `3` note/detik ([configs/local.yaml:L89-95](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/configs/local.yaml#L89-L95)).
   * *Rekomendasi Naskah:* *"Note disaring agar jarak antar ketukan minimal bernilai 200 ms secara global dan 250 ms pada lajur yang sama. Densitas ketukan dibatasi maksimal 3 note per detik."*

7. **Endpoint Riwayat Game dan Leaderboard**
   * *Klaim Skripsi:* Hasil game tidak tersimpan atau leaderboard hanya mockup.
   * *Kondisi Riil:* Endpoint `/api/game-results` dan `/api/game-results/{song_code}/leaderboard` diimplementasikan secara penuh dengan integrasi SQL di tabel `game_results`.
   * *Rekomendasi Naskah:* *"Skor pemain disimpan secara permanen di database dan dikueri secara real-time untuk menyusun leaderboard 10 besar skor tertinggi per lagu."*

---

## 20. Informasi yang Belum Dapat Diverifikasi

* **Data Training Loss per Epoch (Historis):** Karena tidak ada file log teks lengkap atau TensorBoard, riwayat nilai loss training dari epoch 1 s.d 29 tidak dapat diverifikasi secara kronologis. Hanya nilai validation loss dari checkpoint aktif (`best.pt`) pada epoch 30 yang dapat diverifikasi secara pasti (`0.256781`).

---

## 21. Rekomendasi Bukti Gambar dan Tabel untuk Bab IV

Untuk melengkapi visualisasi Bab IV, jalankan perintah dan gunakan file-file yang telah diekspor pada folder [thesis_artifacts/sample_song](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/thesis_artifacts/sample_song):

1. **Gambar Bentuk Waveform Musik:** Visualisasikan bentuk sinyal suara masukan asli menggunakan file gambar `02_waveform.png`.
2. **Gambar Spektrogram:** Tunjukkan perbandingan Spektrogram Mel mentah (`03_mel_spectrogram.png`) vs Log-Mel (`04_log_mel_spectrogram.png`) vs Log-Mel Ternormalisasi (`05_normalized_log_mel_spectrogram.png`).
3. **Tabel Perubahan Dimensi Tensor:** Gunakan tabel alur transformasi tensor pada **Bab II/III** untuk menjelaskan detail penurunan dimensi spektrogram pada CNN.
4. **Tabel Metrik Checkpoint:** Sajikan tabel metrik validasi dari epoch 30 (`best.pt`) sebagai bukti keandalan performa model latih.

---

## 22. Daftar File Sumber yang Diperiksa

1. **FastAPI Entrypoint:** [app/main.py](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/app/main.py)
2. **App Settings:** [app/config.py](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/app/config.py)
3. **Database Setup:** [app/database.py](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/app/database.py)
4. **ORM Models:**
   * [app/models/song.py](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/app/models/song.py)
   * [app/models/beatmap.py](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/app/models/beatmap.py)
   * [app/models/game_result.py](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/app/models/game_result.py)
5. **Routers:**
   * [app/routers/songs.py](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/app/routers/songs.py)
   * [app/routers/beatmaps.py](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/app/routers/beatmaps.py)
   * [app/routers/game_results.py](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/app/routers/game_results.py)
6. **Services:**
   * [app/services/audio_service.py](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/app/services/audio_service.py)
   * [app/services/beatmap_service.py](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/app/services/beatmap_service.py)
   * [app/services/ai_service.py](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/app/services/ai_service.py)
7. **AI Model & Architecture:**
   * [src/beatbert/models/beatmap_model.py](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/src/beatbert/models/beatmap_model.py)
   * [src/beatbert/models/cnn_frontend.py](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/src/beatbert/models/cnn_frontend.py)
   * [src/beatbert/models/transformer.py](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/src/beatbert/models/transformer.py)
8. **Inference & Preprocess Logic:**
   * [src/beatbert/inference/predictor.py](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/src/beatbert/inference/predictor.py)
   * [src/beatbert/inference/postprocess.py](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/src/beatbert/inference/postprocess.py)
   * [src/beatbert/utils/audio.py](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/src/beatbert/utils/audio.py)
   * [src/beatbert/utils/midi.py](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/src/beatbert/utils/midi.py)
9. **Configs & Checkpoints:**
   * [configs/local.yaml](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/configs/local.yaml)
   * [checkpoints/best.pt](file:///d:/Tugas/Semester%207/Skripsi/beatmap_bert_project_real%20-%20Copy/checkpoints/best.pt)
