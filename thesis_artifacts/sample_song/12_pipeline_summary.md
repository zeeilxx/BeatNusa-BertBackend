# Pipeline Execution Summary: malumalu.wav

## Audio Characteristics
* **Original File:** `malumalu.wav`
* **Sampling Rate:** `22050 Hz`
* **Mono/Stereo:** `Mono`
* **Total Duration:** `205.59 seconds (205589 ms)`
* **BPM (Librosa):** `107.67`

## Feature Extraction & Spectrogram
* **Log-Mel Spectrogram Shape:** `[128, 17709] (n_mels = 128, frames = 17709)`
* **Frame Duration:** `11.61 ms`
* **Total Audio Onsets Detected:** `676`
* **Total Audio Beats Detected:** `369`

## Model Configurations
* **Input Dimensions:** `[1, 128, 17709]`
* **d_model:** `128`
* **Transformer Layers:** `2`
* **Attention Heads:** `4`
* **CNN Channels:** `[32, 64, 128]`

## Predictions & Post-Processing Results
* **Raw Events (Confidence >= 0.6):** `598`
* **Postprocessed Notes (After snapping, gap filter, and density limit):** `396`
* **Average Note Density:** `1.93 notes/second`
* **Active Postprocessing Constraints:**
  * **Event Threshold:** `0.6`
  * **Min Gap (Global):** `200 ms`
  * **Same-Lane Min Gap:** `250 ms`
  * **Snap to Onsets:** `True (tolerance: 30 ms)`
  * **Snap to Beats:** `True (tolerance: 40 ms)`
  * **Max Density:** `3 notes/second`
