from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from beatbert.configs import load_config, resolve_paths
from beatbert.inference.predictor import predict_song


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--audio', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cfg = resolve_paths(load_config(args.config), ROOT)
    result = predict_song(args.audio, args.checkpoint, cfg, args.output)
    print(f"Saved {result['num_events']} events to {args.output}")


if __name__ == '__main__':
    main()
