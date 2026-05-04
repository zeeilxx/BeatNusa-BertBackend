from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from beatbert.configs import load_config, resolve_paths
from beatbert.data.preprocess import preprocess_dataset
from beatbert.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = resolve_paths(load_config(args.config), ROOT)
    set_seed(cfg['seed'])
    df = preprocess_dataset(cfg)
    print(df.head())
    print(f'Processed {len(df)} songs -> {cfg["paths"]["processed_dir"]}')


if __name__ == '__main__':
    main()
