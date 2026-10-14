"""Freeze portable image/mask lists without copying or resizing dataset payloads."""
import argparse
import hashlib
import json
from pathlib import Path

from .checkpoint import atomic_json
from .config import load_experiment
from .data import collect_pairs


def build_manifest(cfg, require_test=True):
    root = Path(cfg.data_root).resolve()
    splits = {}
    for split in ("train", "val", "test"):
        try:
            pairs = collect_pairs(cfg, split)
        except FileNotFoundError:
            if split != "test" or require_test:
                raise
            pairs = []
        if not pairs and (split != "test" or require_test):
            raise ValueError(f"Empty {split} split")
        splits[split] = [{"image": image.resolve().relative_to(root).as_posix(),
                          "mask": mask.resolve().relative_to(root).as_posix()}
                         for image, mask in pairs]
        identities = [row["image"] for row in splits[split]]
        if len(identities) != len(set(identities)):
            raise ValueError(f"Duplicate images in {split}")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if {row["image"] for row in splits[a]} & {row["image"] for row in splits[b]}:
            raise ValueError(f"Overlapping {a}/{b} splits")
    identity = json.dumps(splits, sort_keys=True, separators=(",", ":")).encode()
    return {"schema_version": 1, "dataset": cfg.dataset, "split_seed": cfg.split_seed,
            "counts": {k: len(v) for k, v in splits.items()},
            "sha256": hashlib.sha256(identity).hexdigest(),
            "hash_scope": "ordered relative paths, not file contents",
            "splits": splits}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_experiment(args.config, {"data_root": args.data_root})
    payload = build_manifest(cfg)
    path = Path(args.output)
    if path.exists() and json.loads(path.read_text()) != payload:
        raise FileExistsError("Existing manifest differs; choose a new output path")
    atomic_json(payload, path)
    print(json.dumps({"counts": payload["counts"], "sha256": payload["sha256"]}))


if __name__ == "__main__":
    main()
