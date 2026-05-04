"""Quick smoke test: train for 2 epochs to verify everything works."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

from beatbert.configs import load_config, resolve_paths
from beatbert.utils.seed import set_seed

cfg = resolve_paths(load_config(str(ROOT / 'configs' / 'default.yaml')), ROOT)
set_seed(cfg['seed'])

# Override for quick test
cfg['train']['epochs'] = 3
cfg['train']['early_stopping_patience'] = 100  # no early stop for test
cfg['paths']['checkpoint_dir'] = str(ROOT / 'checkpoints_test')

from beatbert.training.trainer import train
result = train(cfg)
print(f"\nSmoke test completed successfully on device: {result.device}")
