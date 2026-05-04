from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


def save_npz(path: str | Path, **arrays: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)  # TANPA kompresi — loading 5-10x lebih cepat saat training


def load_npz(path: str | Path) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def write_json(path: str | Path, data: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
