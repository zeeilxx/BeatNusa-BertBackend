import mido
from pathlib import Path

raw_dir = Path("data/raw")
midi_files = list(raw_dir.glob("*.mid")) + list(raw_dir.glob("*.midi"))

if not midi_files:
    print("Tidak ditemukan file MIDI di folder data/raw/")
    exit()

print(f"Memindai {len(midi_files)} file MIDI...")

unique_pitches = set()

for midi_path in midi_files:
    try:
        midi = mido.MidiFile(midi_path)
        for track in midi.tracks:
            for msg in track:
                if msg.type == 'note_on' and msg.velocity > 0:
                    unique_pitches.add(msg.note)
    except Exception as e:
        print(f"Gagal membaca {midi_path.name}: {e}")

print("\n" + "="*50)
print("HASIL SCAN PITCH MIDI")
print("="*50)
print("Pitch MIDI yang ditemukan di seluruh lagu Anda:")
sorted_pitches = sorted(list(unique_pitches))
print(sorted_pitches)

print("\nBerdasarkan ini, Anda bisa mengelompokkan pitch di atas ke dalam 4 lane di configs/default.yaml")
