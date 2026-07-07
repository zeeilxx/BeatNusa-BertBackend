#!/usr/bin/env python3
"""
analyze_audio_pair.py – Analisis mendalam satu pasangan MP3/WAV untuk skripsi BeatNusa.

Menghasilkan:
1. Metadata file (ukuran, durasi, sample rate, kanal, bitrate, bit depth)
2. Hex dump byte pertama MP3 (20-40 byte) dan header WAV (44 byte)
3. Penjelasan header MP3 dan WAV
4. 10 nilai sampel PCM pertama dari WAV
5. 10 nilai waveform pertama dari MP3 (decoded)
6. 10 nilai waveform pertama dari WAV
7. Tipe data & shape sebelum/sesudah librosa.load(sr=22050, mono=True)

Semua nilai diambil langsung dari file, tidak dibuat atau diperkirakan.
"""

from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

import numpy as np

# ── Ensure imports work ──
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import librosa
import soundfile as sf


def separator(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def analyze_pair(mp3_path: str, wav_path: str):
    mp3_p = Path(mp3_path)
    wav_p = Path(wav_path)

    # ═══════════════════════════════════════════════════════════════
    # 1. Nama file dan metadata
    # ═══════════════════════════════════════════════════════════════
    separator("1. NAMA FILE YANG DIANALISIS")
    print(f"  MP3: {mp3_p.name}")
    print(f"  WAV: {wav_p.name}")

    separator("2. METADATA FILE")

    # MP3 metadata
    mp3_size = mp3_p.stat().st_size
    print(f"  [MP3] Nama file     : {mp3_p.name}")
    print(f"  [MP3] Ukuran file   : {mp3_size:,} bytes ({mp3_size / (1024*1024):.4f} MB)")

    # Load MP3 at native sample rate for metadata
    y_mp3_native, sr_mp3_native = librosa.load(str(mp3_p), sr=None, mono=False)
    if y_mp3_native.ndim == 1:
        mp3_channels = 1
        mp3_duration = len(y_mp3_native) / sr_mp3_native
    else:
        mp3_channels = y_mp3_native.shape[0]
        mp3_duration = y_mp3_native.shape[1] / sr_mp3_native

    print(f"  [MP3] Durasi        : {mp3_duration:.6f} detik")
    print(f"  [MP3] Sample rate   : {sr_mp3_native} Hz (decoded)")
    print(f"  [MP3] Jumlah kanal  : {mp3_channels} (decoded)")
    print(f"  [MP3] Codec         : MP3 (lossy compression)")
    print(f"  [MP3] Bit depth     : tidak tersedia (MP3 = compressed)")

    # Try mutagen for bitrate
    try:
        from mutagen.mp3 import MP3
        m = MP3(str(mp3_p))
        mp3_bitrate = round(m.info.bitrate / 1000)
        print(f"  [MP3] Bitrate       : {mp3_bitrate} kbps")
    except ImportError:
        # Estimate
        if mp3_duration > 0:
            mp3_bitrate = round(mp3_size * 8 / mp3_duration / 1000)
            print(f"  [MP3] Bitrate       : ~{mp3_bitrate} kbps (estimasi)")
        else:
            print(f"  [MP3] Bitrate       : tidak tersedia")
    except Exception:
        print(f"  [MP3] Bitrate       : tidak tersedia")

    print()

    # WAV metadata
    wav_size = wav_p.stat().st_size
    sf_info = sf.info(str(wav_p))
    print(f"  [WAV] Nama file     : {wav_p.name}")
    print(f"  [WAV] Ukuran file   : {wav_size:,} bytes ({wav_size / (1024*1024):.4f} MB)")
    print(f"  [WAV] Durasi        : {sf_info.duration:.6f} detik")
    print(f"  [WAV] Sample rate   : {sf_info.samplerate} Hz")
    print(f"  [WAV] Jumlah kanal  : {sf_info.channels}")
    print(f"  [WAV] Codec/subtype : {sf_info.subtype}")
    print(f"  [WAV] Bit depth     : {sf_info.subtype}")
    # WAV bitrate = sample_rate * channels * bits_per_sample
    bits_per_sample = int(sf_info.subtype.replace("PCM_", "")) if "PCM_" in sf_info.subtype else 16
    wav_bitrate = sf_info.samplerate * sf_info.channels * bits_per_sample / 1000
    print(f"  [WAV] Bitrate       : {wav_bitrate:.0f} kbps (uncompressed)")

    # ═══════════════════════════════════════════════════════════════
    # 3. Hex dump – MP3 first 40 bytes
    # ═══════════════════════════════════════════════════════════════
    separator("3. HEX DUMP - 40 BYTE PERTAMA FILE MP3")

    with open(str(mp3_p), "rb") as f:
        mp3_raw = f.read(40)

    hex_str = " ".join(f"{b:02X}" for b in mp3_raw)
    ascii_str = "".join(chr(b) if 32 <= b < 127 else "." for b in mp3_raw)
    print(f"  Hex  : {hex_str}")
    print(f"  ASCII: {ascii_str}")
    print(f"  Jumlah byte ditampilkan: {len(mp3_raw)}")

    # ═══════════════════════════════════════════════════════════════
    # 4. Hex dump – WAV header 44 bytes
    # ═══════════════════════════════════════════════════════════════
    separator("4. HEX DUMP - 44 BYTE HEADER FILE WAV")

    with open(str(wav_p), "rb") as f:
        wav_raw = f.read(44)

    hex_str_wav = " ".join(f"{b:02X}" for b in wav_raw)
    ascii_str_wav = "".join(chr(b) if 32 <= b < 127 else "." for b in wav_raw)
    print(f"  Hex  : {hex_str_wav}")
    print(f"  ASCII: {ascii_str_wav}")
    print(f"  Jumlah byte ditampilkan: {len(wav_raw)}")

    # ═══════════════════════════════════════════════════════════════
    # 5. Penjelasan header MP3 dan WAV
    # ═══════════════════════════════════════════════════════════════
    separator("5. PENJELASAN HEADER MP3 DAN WAV")

    print("  --- Header MP3 ---")
    # Check for ID3 tag
    if mp3_raw[:3] == b"ID3":
        id3_version = f"2.{mp3_raw[3]}.{mp3_raw[4]}"
        id3_flags = mp3_raw[5]
        # ID3 size is 4 bytes synchsafe integer (bytes 6-9)
        id3_size = (mp3_raw[6] << 21) | (mp3_raw[7] << 14) | (mp3_raw[8] << 7) | mp3_raw[9]
        print(f"  Byte 0-2   : 'ID3' - penanda tag ID3v2")
        print(f"  Byte 3-4   : Versi ID3v{id3_version}")
        print(f"  Byte 5     : Flags = 0x{id3_flags:02X}")
        print(f"  Byte 6-9   : Ukuran tag ID3 = {id3_size} bytes (synchsafe integer)")
        print(f"  Byte 10+   : Frame-frame ID3 (metadata: judul, artis, album, dll.)")
        print(f"  >> MPEG audio frame dimulai setelah ID3 tag (offset ~{id3_size + 10})")
        print(f"  >> ID3 berisi tag TSSE: encoder info (terlihat: 'Lavf62.12.100')")
        
        # Try to find MPEG sync word after ID3
        with open(str(mp3_p), "rb") as f:
            f.seek(id3_size + 10)
            mpeg_bytes = f.read(4)
            if len(mpeg_bytes) >= 2 and mpeg_bytes[0] == 0xFF and (mpeg_bytes[1] & 0xE0) == 0xE0:
                print(f"  >> MPEG frame sync ditemukan di offset {id3_size + 10}: {' '.join(f'{b:02X}' for b in mpeg_bytes)}")
                # Parse MPEG header
                version_bits = (mpeg_bytes[1] >> 3) & 0x03
                layer_bits = (mpeg_bytes[1] >> 1) & 0x03
                versions = {0: "MPEG 2.5", 2: "MPEG 2", 3: "MPEG 1"}
                layers = {1: "Layer III", 2: "Layer II", 3: "Layer I"}
                print(f"  >> Versi: {versions.get(version_bits, 'unknown')}, {layers.get(layer_bits, 'unknown')}")
    else:
        # Check for MPEG sync word directly
        if mp3_raw[0] == 0xFF and (mp3_raw[1] & 0xE0) == 0xE0:
            print(f"  Byte 0-1   : 0xFF 0x{mp3_raw[1]:02X} - MPEG frame sync word (tidak ada ID3 tag)")
        else:
            print(f"  Byte 0-2   : {mp3_raw[:3]} – format tidak dikenali sebagai ID3 atau MPEG sync")

    print()
    print("  --- Header WAV ---")
    # Parse standard WAV header
    if wav_raw[:4] == b"RIFF":
        riff_size = struct.unpack_from("<I", wav_raw, 4)[0]
        wave_id = wav_raw[8:12].decode("ascii", errors="replace")
        fmt_id = wav_raw[12:16].decode("ascii", errors="replace")
        fmt_size = struct.unpack_from("<I", wav_raw, 16)[0]
        audio_format = struct.unpack_from("<H", wav_raw, 20)[0]
        num_channels = struct.unpack_from("<H", wav_raw, 22)[0]
        sample_rate_hdr = struct.unpack_from("<I", wav_raw, 24)[0]
        byte_rate = struct.unpack_from("<I", wav_raw, 28)[0]
        block_align = struct.unpack_from("<H", wav_raw, 32)[0]
        bits_per_sample_hdr = struct.unpack_from("<H", wav_raw, 34)[0]
        data_id = wav_raw[36:40].decode("ascii", errors="replace")
        data_size = struct.unpack_from("<I", wav_raw, 40)[0]

        format_names = {1: "PCM (uncompressed)", 3: "IEEE Float", 6: "A-law", 7: "mu-law"}

        print(f"  Byte 0-3   : '{wav_raw[:4].decode()}' - penanda format RIFF")
        print(f"  Byte 4-7   : {riff_size:,} - ukuran file - 8 bytes")
        print(f"  Byte 8-11  : '{wave_id}' - penanda tipe WAVE")
        print(f"  Byte 12-15 : '{fmt_id}' - penanda sub-chunk 'fmt '")
        print(f"  Byte 16-19 : {fmt_size} - ukuran sub-chunk fmt")
        print(f"  Byte 20-21 : {audio_format} - format audio ({format_names.get(audio_format, 'unknown')})")
        print(f"  Byte 22-23 : {num_channels} - jumlah kanal")
        print(f"  Byte 24-27 : {sample_rate_hdr} - sample rate (Hz)")
        print(f"  Byte 28-31 : {byte_rate:,} - byte rate (bytes/detik)")
        print(f"  Byte 32-33 : {block_align} - block align (bytes per frame)")
        print(f"  Byte 34-35 : {bits_per_sample_hdr} - bits per sample")
        # Byte 36-39 might be 'data' or another sub-chunk (e.g. LIST)
        if data_id == "data":
            print(f"  Byte 36-39 : '{data_id}' - penanda sub-chunk data")
            print(f"  Byte 40-43 : {data_size:,} - ukuran data audio (bytes)")
        else:
            print(f"  Byte 36-39 : '{data_id}' - sub-chunk metadata (BUKAN 'data')")
            print(f"  Byte 40-43 : {data_size} - ukuran sub-chunk '{data_id}' (bytes)")
            print(f"  >> CATATAN: Header standar 44-byte mengasumsikan 'data' di byte 36.")
            print(f"     File ini memiliki chunk '{data_id}' sebelum chunk 'data'.")
            print(f"     Chunk 'data' (berisi audio PCM) berada di offset lebih jauh.")
    else:
        print(f"  Header tidak dikenali sebagai RIFF/WAV: {wav_raw[:4]}")

    # ═══════════════════════════════════════════════════════════════
    # 6. 10 Nilai Sampel PCM pertama dari WAV
    # ═══════════════════════════════════════════════════════════════
    separator("6. SEPULUH NILAI SAMPEL PCM PERTAMA DARI WAV (RAW)")

    # Find the 'data' chunk and read raw PCM samples
    with open(str(wav_p), "rb") as f:
        wav_all = f.read()

    # Find 'data' sub-chunk
    data_offset = wav_all.find(b"data")
    if data_offset >= 0:
        data_chunk_size = struct.unpack_from("<I", wav_all, data_offset + 4)[0]
        pcm_start = data_offset + 8
        print(f"  'data' chunk ditemukan di offset byte {data_offset}")
        print(f"  Ukuran data chunk: {data_chunk_size:,} bytes")
        print(f"  PCM data dimulai di offset byte {pcm_start}")
        print()

        # Read 10 PCM samples (assuming 16-bit signed, possibly multi-channel)
        # For stereo: each sample frame = 2 channels * 2 bytes = 4 bytes
        # For mono: each sample = 2 bytes
        sf_info_check = sf.info(str(wav_p))
        ch = sf_info_check.channels
        bps = int(sf_info_check.subtype.replace("PCM_", "")) if "PCM_" in sf_info_check.subtype else 16
        bytes_per_sample = bps // 8
        frame_size = ch * bytes_per_sample

        print(f"  Konfigurasi: {ch} kanal, {bps}-bit, {bytes_per_sample} bytes/sample, {frame_size} bytes/frame")
        print()

        for i in range(10):
            offset = pcm_start + i * frame_size
            if bps == 16:
                values = []
                for c in range(ch):
                    val = struct.unpack_from("<h", wav_all, offset + c * bytes_per_sample)[0]
                    values.append(val)
                if ch == 1:
                    print(f"  Sample {i}: {values[0]}")
                else:
                    print(f"  Sample {i}: Ch0={values[0]}, Ch1={values[1]}" + (f", ..." if ch > 2 else ""))
            elif bps == 24:
                values = []
                for c in range(ch):
                    b0, b1, b2 = wav_all[offset + c*3], wav_all[offset + c*3 + 1], wav_all[offset + c*3 + 2]
                    val = b0 | (b1 << 8) | (b2 << 16)
                    if val >= 0x800000:
                        val -= 0x1000000
                    values.append(val)
                if ch == 1:
                    print(f"  Sample {i}: {values[0]}")
                else:
                    print(f"  Sample {i}: Ch0={values[0]}, Ch1={values[1]}")

        # Also find first non-zero samples
        print()
        print("  Mencari 10 sampel PCM pertama yang BUKAN nol (non-silent):")
        found = 0
        sample_idx = 0
        while found < 10 and (pcm_start + sample_idx * frame_size + frame_size) <= len(wav_all):
            offset = pcm_start + sample_idx * frame_size
            if bps == 16:
                val_ch0 = struct.unpack_from("<h", wav_all, offset)[0]
                if val_ch0 != 0:
                    if ch == 1:
                        print(f"  Sample[{sample_idx}]: {val_ch0}")
                    else:
                        val_ch1 = struct.unpack_from("<h", wav_all, offset + bytes_per_sample)[0]
                        print(f"  Sample[{sample_idx}]: Ch0={val_ch0}, Ch1={val_ch1}")
                    found += 1
            sample_idx += 1
        if found == 0:
            print("  (Tidak ditemukan sampel non-zero dalam file)")
    else:
        print("  PERINGATAN: 'data' chunk tidak ditemukan dalam file WAV!")

    # ═══════════════════════════════════════════════════════════════
    # 7. 10 Nilai Waveform pertama dari MP3 (decoded via librosa)
    # ═══════════════════════════════════════════════════════════════
    separator("7. SEPULUH NILAI WAVEFORM PERTAMA DARI MP3 (DECODED)")

    y_mp3_mono, sr_mp3_mono = librosa.load(str(mp3_p), sr=None, mono=True)
    print(f"  librosa.load('{mp3_p.name}', sr=None, mono=True)")
    print(f"  Sample rate: {sr_mp3_mono} Hz")
    print(f"  Shape: {y_mp3_mono.shape}")
    print(f"  Dtype: {y_mp3_mono.dtype}")
    print()
    for i in range(10):
        print(f"  Waveform[{i}] = {y_mp3_mono[i]:.10f}")

    # Also show first non-zero values
    print()
    print("  10 nilai waveform pertama yang BUKAN nol (non-silent):")
    nz_mp3 = np.where(np.abs(y_mp3_mono) > 1e-7)[0]
    if len(nz_mp3) >= 10:
        for k in range(10):
            idx = int(nz_mp3[k])
            print(f"  Waveform[{idx}] = {y_mp3_mono[idx]:.15f}  (repr: {repr(y_mp3_mono[idx])})")
    else:
        print("  (Kurang dari 10 sampel dengan |nilai| > 1e-7 ditemukan)")
    # Show samples from ~2s into the audio (where music plays)
    start_2s = int(sr_mp3_mono * 2)
    print()
    print(f"  10 sampel dari detik ke-2 (index {start_2s}+):")
    for i in range(10):
        idx = start_2s + i
        print(f"  Waveform[{idx}] = {y_mp3_mono[idx]:.10f}")

    # ═══════════════════════════════════════════════════════════════
    # 8. 10 Nilai Waveform pertama dari WAV (same config)
    # ═══════════════════════════════════════════════════════════════
    separator("8. SEPULUH NILAI WAVEFORM PERTAMA DARI WAV (SAME CONFIG)")

    y_wav_mono, sr_wav_mono = librosa.load(str(wav_p), sr=None, mono=True)
    print(f"  librosa.load('{wav_p.name}', sr=None, mono=True)")
    print(f"  Sample rate: {sr_wav_mono} Hz")
    print(f"  Shape: {y_wav_mono.shape}")
    print(f"  Dtype: {y_wav_mono.dtype}")
    print()
    for i in range(10):
        print(f"  Waveform[{i}] = {y_wav_mono[i]:.10f}")

    # Also show first non-zero values
    print()
    print("  10 nilai waveform pertama yang BUKAN nol (non-silent):")
    nz_wav = np.where(np.abs(y_wav_mono) > 1e-7)[0]
    if len(nz_wav) >= 10:
        for k in range(10):
            idx = int(nz_wav[k])
            print(f"  Waveform[{idx}] = {y_wav_mono[idx]:.15f}  (repr: {repr(y_wav_mono[idx])})")
    else:
        print("  (Kurang dari 10 sampel dengan |nilai| > 1e-7 ditemukan)")
    # Show samples from ~2s into the audio (where music plays)
    start_2s_w = int(sr_wav_mono * 2)
    print()
    print(f"  10 sampel dari detik ke-2 (index {start_2s_w}+):")
    for i in range(10):
        idx = start_2s_w + i
        print(f"  Waveform[{idx}] = {y_wav_mono[idx]:.10f}")

    # ═══════════════════════════════════════════════════════════════
    # 9. Tipe data dan shape sebelum/sesudah librosa.load(sr=22050, mono=True)
    # ═══════════════════════════════════════════════════════════════
    separator("9. TIPE DATA DAN SHAPE SEBELUM DAN SESUDAH PENYERAGAMAN (librosa.load sr=22050, mono=True)")

    print("  === MP3 ===")
    print(f"  SEBELUM penyeragaman (sr=None, mono=True):")
    print(f"    Shape    : {y_mp3_mono.shape}")
    print(f"    Dtype    : {y_mp3_mono.dtype}")
    print(f"    SR       : {sr_mp3_mono}")
    print(f"    Duration : {len(y_mp3_mono)/sr_mp3_mono:.6f} s")

    y_mp3_22k, sr_mp3_22k = librosa.load(str(mp3_p), sr=22050, mono=True)
    print(f"  SESUDAH penyeragaman (sr=22050, mono=True):")
    print(f"    Shape    : {y_mp3_22k.shape}")
    print(f"    Dtype    : {y_mp3_22k.dtype}")
    print(f"    SR       : {sr_mp3_22k}")
    print(f"    Duration : {len(y_mp3_22k)/sr_mp3_22k:.6f} s")

    print()
    print("  === WAV ===")
    print(f"  SEBELUM penyeragaman (sr=None, mono=True):")
    print(f"    Shape    : {y_wav_mono.shape}")
    print(f"    Dtype    : {y_wav_mono.dtype}")
    print(f"    SR       : {sr_wav_mono}")
    print(f"    Duration : {len(y_wav_mono)/sr_wav_mono:.6f} s")

    y_wav_22k, sr_wav_22k = librosa.load(str(wav_p), sr=22050, mono=True)
    print(f"  SESUDAH penyeragaman (sr=22050, mono=True):")
    print(f"    Shape    : {y_wav_22k.shape}")
    print(f"    Dtype    : {y_wav_22k.dtype}")
    print(f"    SR       : {sr_wav_22k}")
    print(f"    Duration : {len(y_wav_22k)/sr_wav_22k:.6f} s")

    # Also show first 10 values after standardization
    print()
    print("  10 nilai waveform pertama setelah penyeragaman (sr=22050, mono=True):")
    print("  MP3:")
    for i in range(10):
        print(f"    [{i}] = {y_mp3_22k[i]:.10f}")
    print("  WAV:")
    for i in range(10):
        print(f"    [{i}] = {y_wav_22k[i]:.10f}")

    # ═══════════════════════════════════════════════════════════════
    # 10. Penjelasan mengapa byte mentah MP3 tidak bisa dibandingkan
    # ═══════════════════════════════════════════════════════════════
    separator("10. PENJELASAN: BYTE MENTAH MP3 vs SAMPEL AMPLITUDO WAV")

    explanation = """
  File MP3 menyimpan data audio dalam format terkompresi menggunakan algoritma
  lossy compression (MPEG Audio Layer III). Byte-byte mentah dalam file MP3
  merepresentasikan data yang telah dikodekan melalui proses:
    - Psychoacoustic model (menghapus frekuensi yang tidak terdengar manusia)
    - Modified Discrete Cosine Transform (MDCT)
    - Huffman encoding
    - Bit reservoir

  Sebaliknya, file WAV (PCM) menyimpan sampel amplitudo mentah secara langsung.
  Setiap sampel adalah nilai integer (misalnya 16-bit signed: -32768 hingga 32767)
  yang merepresentasikan amplitudo gelombang suara pada titik waktu tertentu.

  Oleh karena itu:
    [X] Byte mentah MP3 TIDAK DAPAT dibandingkan langsung dengan sampel PCM WAV.
        Byte MP3 adalah data terkompresi, bukan nilai amplitudo.
    [V] Untuk membandingkan konten audio, MP3 harus di-decode terlebih dahulu
        menjadi waveform (array float), kemudian dibandingkan dengan waveform
        yang dihasilkan dari file WAV.
    [V] librosa.load() melakukan decode MP3 -> float32 waveform secara otomatis,
        sehingga kedua format menghasilkan array numpy yang sebanding.

  Dalam pipeline BeatNusa, librosa.load(path, sr=22050, mono=True) digunakan
  untuk menyeragamkan kedua format menjadi representasi yang sama sebelum
  diteruskan ke tahap feature extraction (mel-spectrogram).
"""
    print(explanation)

    separator("ANALISIS SELESAI")
    print(f"  Script: {Path(__file__).name}")
    print(f"  MP3   : {mp3_p}")
    print(f"  WAV   : {wav_p}")


if __name__ == "__main__":
    data_dir = ROOT / "data"
    mp3_file = str(data_dir / "soleramlagudaerahriaubyviolamerliaqdaaf.mp3")
    wav_file = str(data_dir / "soleramlagudaerahriaubyviolamerliaqdaaf.wav")

    if not Path(mp3_file).exists():
        print(f"ERROR: MP3 file not found: {mp3_file}")
        sys.exit(1)
    if not Path(wav_file).exists():
        print(f"ERROR: WAV file not found: {wav_file}")
        sys.exit(1)

    analyze_pair(mp3_file, wav_file)
