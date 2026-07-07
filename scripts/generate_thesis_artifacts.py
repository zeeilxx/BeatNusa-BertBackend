import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure 'src' is in the python path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from beatbert.configs import load_config, resolve_paths
from beatbert.models.beatmap_model import BeatmapModel
from beatbert.utils.audio import extract_audio_features, frame_times_ms
from beatbert.inference.predictor import _run_chunked_inference
from beatbert.inference.postprocess import postprocess_events

def generate_artifacts(audio_path_str: str, config_path_str: str, checkpoint_path_str: str, output_dir_str: str):
    audio_path = Path(audio_path_str)
    config_path = Path(config_path_str)
    checkpoint_path = Path(checkpoint_path_str)
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading config from {config_path}...")
    cfg = resolve_paths(load_config(str(config_path)), ROOT)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading model and checkpoint...")
    model = BeatmapModel(cfg).to(device)
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    print(f"Extracting features from {audio_path}...")
    import librosa
    # Load audio natively for metadata
    y, sr = librosa.load(str(audio_path), sr=cfg['audio']['sample_rate'], mono=cfg['audio'].get('mono', True))
    duration_sec = len(y) / sr
    
    features = extract_audio_features(str(audio_path), cfg)
    mel = features.mel  # Normalized Log-Mel Spectrogram, shape: [128, T]
    T = mel.shape[1]
    f_ms = frame_times_ms(T, cfg)
    
    # 01. Audio Metadata
    metadata = {
        "song_name": audio_path.name,
        "sample_rate": sr,
        "mono": bool(cfg['audio'].get('mono', True)),
        "duration_seconds": duration_sec,
        "duration_ms": duration_sec * 1000.0,
        "num_frames": T,
        "bpm": features.bpm,
        "detected_onsets_count": len(features.onset_times_ms),
        "detected_beats_count": len(features.beat_times_ms)
    }
    with open(output_dir / "01_audio_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("Saved 01_audio_metadata.json")
    
    # Plotting waveforms and spectrograms using matplotlib
    try:
        import matplotlib.pyplot as plt
        
        # 02. Waveform
        plt.figure(figsize=(10, 3))
        plt.plot(np.linspace(0, duration_sec, len(y)), y, color='blue', alpha=0.6)
        plt.title(f"Waveform - {audio_path.name}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.savefig(output_dir / "02_waveform.png", dpi=150)
        plt.close()
        print("Saved 02_waveform.png")
        
        # 03. Raw Mel Spectrogram
        # Re-compute raw mel (before log and normalization)
        raw_mel = librosa.feature.melspectrogram(
            y=y, sr=sr,
            n_fft=cfg['audio']['n_fft'],
            hop_length=cfg['audio']['hop_length'],
            win_length=cfg['audio']['win_length'],
            n_mels=cfg['audio']['n_mels'],
            fmin=cfg['audio']['fmin'],
            fmax=cfg['audio']['fmax'],
            power=2.0
        )
        plt.figure(figsize=(10, 3))
        librosa.display.specshow(raw_mel, sr=sr, hop_length=cfg['audio']['hop_length'], x_axis='time', y_axis='mel', fmin=cfg['audio']['fmin'], fmax=cfg['audio']['fmax'])
        plt.colorbar(format='%+2.0f dB')
        plt.title("Raw Mel Spectrogram")
        plt.tight_layout()
        plt.savefig(output_dir / "03_mel_spectrogram.png", dpi=150)
        plt.close()
        print("Saved 03_mel_spectrogram.png")
        
        # 04. Log-Mel Spectrogram (dB)
        log_mel = librosa.power_to_db(raw_mel, ref=np.max, top_db=cfg['audio'].get('top_db', 80))
        plt.figure(figsize=(10, 3))
        librosa.display.specshow(log_mel, sr=sr, hop_length=cfg['audio']['hop_length'], x_axis='time', y_axis='mel', fmin=cfg['audio']['fmin'], fmax=cfg['audio']['fmax'])
        plt.colorbar(format='%+2.0f dB')
        plt.title("Log-Mel Spectrogram (dB)")
        plt.tight_layout()
        plt.savefig(output_dir / "04_log_mel_spectrogram.png", dpi=150)
        plt.close()
        print("Saved 04_log_mel_spectrogram.png")
        
        # 05. Normalized Log-Mel Spectrogram
        plt.figure(figsize=(10, 3))
        librosa.display.specshow(mel, sr=sr, hop_length=cfg['audio']['hop_length'], x_axis='time', y_axis='mel', fmin=cfg['audio']['fmin'], fmax=cfg['audio']['fmax'])
        plt.colorbar()
        plt.title("Normalized Log-Mel Spectrogram (Zero Mean, Unit Variance)")
        plt.tight_layout()
        plt.savefig(output_dir / "05_normalized_log_mel_spectrogram.png", dpi=150)
        plt.close()
        print("Saved 05_normalized_log_mel_spectrogram.png")
        
    except Exception as plterr:
        print(f"Warning: Failed to generate plots using matplotlib: {plterr}")
        
    # Running model to extract intermediate vectors (we take a slice/context of 256 frames from the middle)
    print("Extracting intermediate vectors...")
    with torch.no_grad():
        # Let's take a 256-frame slice from the middle of the song
        start_frame = max(0, T // 2 - 128)
        end_frame = start_frame + 256
        mel_slice = mel[:, start_frame:end_frame]
        valid_len = mel_slice.shape[1]
        
        if valid_len < 256:
            # Pad
            mel_slice = np.pad(mel_slice, ((0, 0), (0, 256 - valid_len)))
            mask_np = np.concatenate([np.ones(valid_len, dtype=np.float32), np.zeros(256 - valid_len, dtype=np.float32)])
        else:
            mask_np = np.ones(256, dtype=np.float32)
            
        x_tensor = torch.from_numpy(mel_slice).unsqueeze(0).to(device)  # [1, 128, 256]
        
        # CNN frontend forward
        cnn_out = model.frontend(x_tensor)  # [1, 256, d_model]
        # Positional embedding & encoder forward
        pos_out = model.positional(cnn_out)
        trans_out = model.encoder(pos_out)  # [1, 256, d_model]
        
        cnn_np = cnn_out.squeeze(0).cpu().numpy()  # [256, 128]
        trans_np = trans_out.squeeze(0).cpu().numpy()  # [256, 128]
        
        # 06. CNN Vector Stats and Samples
        cnn_vector_info = {
            "tensor_shape": list(cnn_out.shape),
            "stats": {
                "min": float(np.min(cnn_np)),
                "max": float(np.max(cnn_np)),
                "mean": float(np.mean(cnn_np)),
                "std": float(np.std(cnn_np))
            },
            "sample_frames_count": 5,
            "sample_frames": [
                {
                    "frame_index": int(start_frame + i * 50),
                    "first_20_values": [float(val) for val in cnn_np[i * 50, :20]]
                }
                for i in range(min(5, len(cnn_np) // 50 + 1))
            ]
        }
        with open(output_dir / "06_cnn_vector.json", "w") as f:
            json.dump(cnn_vector_info, f, indent=2)
        print("Saved 06_cnn_vector.json")
        
        # 07. Transformer Vector Stats and Samples
        trans_vector_info = {
            "tensor_shape": list(trans_out.shape),
            "stats": {
                "min": float(np.min(trans_np)),
                "max": float(np.max(trans_np)),
                "mean": float(np.mean(trans_np)),
                "std": float(np.std(trans_np))
            },
            "sample_frames_count": 5,
            "sample_frames": [
                {
                    "frame_index": int(start_frame + i * 50),
                    "first_20_values": [float(val) for val in trans_np[i * 50, :20]]
                }
                for i in range(min(5, len(trans_np) // 50 + 1))
            ]
        }
        with open(output_dir / "07_transformer_vector.json", "w") as f:
            json.dump(trans_vector_info, f, indent=2)
        print("Saved 07_transformer_vector.json")

    # Run full chunked inference for predictions
    print("Running chunked inference...")
    event_prob, lane_prob = _run_chunked_inference(model, mel, cfg, device)
    lane_pred = lane_prob.argmax(axis=-1)
    
    # 08 & 09. Raw Predictions (thresholded)
    threshold = cfg['postprocess']['event_threshold']
    raw_events = []
    idxs = np.where(event_prob >= threshold)[0]
    for idx in idxs:
        raw_events.append({
            "frame_idx": int(idx),
            "time_ms": float(f_ms[idx]),
            "confidence": float(event_prob[idx]),
            "predicted_lane": int(lane_pred[idx])
        })
        
    df_raw_events = pd.DataFrame(raw_events)
    if not df_raw_events.empty:
        df_raw_events.to_csv(output_dir / "08_raw_event_predictions.csv", index=False)
        
        # Save raw lane logits/probabilities for those frames
        raw_lanes = []
        for idx in idxs:
            raw_lanes.append({
                "frame_idx": int(idx),
                "time_ms": float(f_ms[idx]),
                "lane_0_prob": float(lane_prob[idx, 0]),
                "lane_1_prob": float(lane_prob[idx, 1]),
                "lane_2_prob": float(lane_prob[idx, 2]),
                "lane_3_prob": float(lane_prob[idx, 3]),
                "argmax_lane": int(lane_pred[idx])
            })
        pd.DataFrame(raw_lanes).to_csv(output_dir / "09_raw_lane_predictions.csv", index=False)
    else:
        # Create empty CSVs
        pd.DataFrame(columns=["frame_idx", "time_ms", "confidence", "predicted_lane"]).to_csv(output_dir / "08_raw_event_predictions.csv", index=False)
        pd.DataFrame(columns=["frame_idx", "time_ms", "lane_0_prob", "lane_1_prob", "lane_2_prob", "lane_3_prob", "argmax_lane"]).to_csv(output_dir / "09_raw_lane_predictions.csv", index=False)
    print("Saved 08_raw_event_predictions.csv & 09_raw_lane_predictions.csv")
    
    # 10. Postprocessed Notes
    print("Running postprocessing...")
    raw_events_post = [
        {"time_ms": e["time_ms"], "lane": e["predicted_lane"], "confidence": e["confidence"]}
        for e in raw_events
    ]
    final_events = postprocess_events(raw_events_post, features.onset_times_ms, features.beat_times_ms, cfg)
    
    df_final_events = pd.DataFrame([
        {
            "note_idx": i,
            "time_ms": int(round(e["time_ms"])),
            "lane": int(e["lane"]),
            "confidence": float(e["confidence"])
        }
        for i, e in enumerate(final_events)
    ])
    df_final_events.to_csv(output_dir / "10_postprocessed_notes.csv", index=False)
    print("Saved 10_postprocessed_notes.csv")
    
    # 11. Final Beatmap JSON
    notes_payload = [
        {
            "time_ms": int(round(e["time_ms"])),
            "lane": int(e["lane"]),
            "type": "tap",
            "length_ms": 0
        }
        for e in final_events
    ]
    beatmap_json = {
        "lane_count": int(cfg['model']['num_lanes']),
        "offset_ms": 0,
        "notes": notes_payload
    }
    with open(output_dir / "11_final_beatmap.json", "w") as f:
        json.dump(beatmap_json, f, indent=2)
    print("Saved 11_final_beatmap.json")
    
    # 12. Pipeline Summary (Markdown)
    summary_md = f"""# Pipeline Execution Summary: {audio_path.name}

## Audio Characteristics
* **Original File:** `{audio_path.name}`
* **Sampling Rate:** `{sr} Hz`
* **Mono/Stereo:** `{"Mono" if cfg['audio'].get('mono', True) else "Stereo"}`
* **Total Duration:** `{duration_sec:.2f} seconds ({duration_sec * 1000.0:.0f} ms)`
* **BPM (Librosa):** `{features.bpm:.2f}`

## Feature Extraction & Spectrogram
* **Log-Mel Spectrogram Shape:** `{list(features.mel.shape)} (n_mels = {cfg['audio']['n_mels']}, frames = {features.mel.shape[1]})`
* **Frame Duration:** `{cfg['audio']['hop_length'] / cfg['audio']['sample_rate'] * 1000.0:.2f} ms`
* **Total Audio Onsets Detected:** `{len(features.onset_times_ms)}`
* **Total Audio Beats Detected:** `{len(features.beat_times_ms)}`

## Model Configurations
* **Input Dimensions:** `[1, {cfg['model']['input_mels']}, {T}]`
* **d_model:** `{cfg['model']['d_model']}`
* **Transformer Layers:** `{cfg['model']['num_layers']}`
* **Attention Heads:** `{cfg['model']['num_heads']}`
* **CNN Channels:** `{cfg['model']['cnn_channels']}`

## Predictions & Post-Processing Results
* **Raw Events (Confidence >= {threshold}):** `{len(raw_events)}`
* **Postprocessed Notes (After snapping, gap filter, and density limit):** `{len(notes_payload)}`
* **Average Note Density:** `{len(notes_payload) / duration_sec:.2f} notes/second`
* **Active Postprocessing Constraints:**
  * **Event Threshold:** `{cfg['postprocess']['event_threshold']}`
  * **Min Gap (Global):** `{cfg['postprocess']['min_gap_ms']} ms`
  * **Same-Lane Min Gap:** `{cfg['postprocess']['same_lane_min_gap_ms']} ms`
  * **Snap to Onsets:** `{cfg['postprocess']['snap_to_onsets']} (tolerance: {cfg['postprocess']['onset_snap_tolerance_ms']} ms)`
  * **Snap to Beats:** `{cfg['postprocess']['snap_to_beats']} (tolerance: {cfg['postprocess']['beat_snap_tolerance_ms']} ms)`
  * **Max Density:** `{cfg['postprocess'].get('max_density_notes_per_second', 8)} notes/second`
"""
    with open(output_dir / "12_pipeline_summary.md", "w") as f:
        f.write(summary_md)
    print("Saved 12_pipeline_summary.md")
    print("All thesis artifacts successfully generated!")

if __name__ == '__main__':
    generate_artifacts(
        audio_path_str="data/predictsongs/malumalu.wav",
        config_path_str="configs/local.yaml",
        checkpoint_path_str="checkpoints/best.pt",
        output_dir_str="thesis_artifacts/sample_song"
    )
