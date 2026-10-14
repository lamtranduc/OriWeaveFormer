"""Checkpoint IO with atomic replacement and explicit EMA selection."""
import hashlib
import json
import os
from pathlib import Path

import torch


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def extract_weights(saved, prefer_ema=True):
    if not isinstance(saved, dict):
        raise ValueError("Expected a checkpoint dictionary")
    ordered = ("ema", "model", "state_dict") if prefer_ema else ("model", "state_dict")
    for key in ordered:
        state = saved.get(key)
        if isinstance(state, dict) and state and all(torch.is_tensor(v) for v in state.values()):
            return {k.removeprefix("module."): v for k, v in state.items()}, key
    if saved and all(torch.is_tensor(v) for v in saved.values()):
        return {k.removeprefix("module."): v for k, v in saved.items()}, "state_dict"
    raise ValueError(f"No usable {'EMA/model' if prefer_ema else 'model'} weights in checkpoint")
