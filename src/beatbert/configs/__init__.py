from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def resolve_paths(config: Dict[str, Any], project_root: str | Path) -> Dict[str, Any]:
    root = Path(project_root)
    cfg = dict(config)
    cfg['project_root'] = str(root)
    cfg['paths'] = dict(config['paths'])
    for key, value in cfg['paths'].items():
        cfg['paths'][key] = str((root / value).resolve())
        Path(cfg['paths'][key]).mkdir(parents=True, exist_ok=True)
    return cfg
