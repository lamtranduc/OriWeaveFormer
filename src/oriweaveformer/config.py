"""Strict YAML composition; paths in defaults are relative to their YAML file."""
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


@dataclass
class Config:
    experiment_id: str = "oriweaveformer_final"
    dataset: str = "deepglobe"
    data_root: str = "data/deepglobe/raw"
    output_dir: str = "runs/oriweaveformer_final/deepglobe"
    encoder: str = "mit_b3"
    pretrained: str | None = "imagenet"
    crop_size: int = 512
    epochs: int = 150
    schedule_epochs: int | None = None
    batch_size: int = 8  # PER PROCESS; global = this * WORLD_SIZE.
    accumulation_steps: int = 1
    optimizer: str = "adamw"
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    ema_decay: float = 0.999
    amp: str = "float16"
    num_workers: int = 6
    val_workers: int = 2
    persistent_workers: bool = False
    val_interval: int = 10
    seed: int = 42
    split_seed: int = 42
    split_manifest: str | None = None
    val_fraction: float = 0.1
    road_crop_probability: float = 0.7
    min_road_fraction: float = 0.01
    decoder_dim: int = 256
    decoder_dropout: float = 0.1
    spans: tuple[int, int, int] = (4, 3, 8)
    ocfe_initial_scale: float = 0.1
    loss: str = "bce_dice"
    init_checkpoint: str | None = None
    resume: bool = False
    full_checkpoint_interval: int = 10


def _read_yaml(path, stack=()):
    path = Path(path).resolve()
    if path in stack:
        raise ValueError(f"Cyclic YAML defaults: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    defaults = payload.pop("defaults", [])
    if not isinstance(defaults, list):
        raise ValueError("defaults must be a list of relative YAML paths")
    result = {}
    for default in defaults:
        result.update(_read_yaml(path.parent / default, (*stack, path)))
    result.update(payload)
    return result


def load_experiment(path, overrides=None):
    payload = _read_yaml(path)
    payload.update({k: v for k, v in (overrides or {}).items() if v is not None})
    known = {field.name for field in fields(Config)}
    if unknown := set(payload) - known:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    if "spans" in payload:
        payload["spans"] = tuple(payload["spans"])
    cfg = Config(**payload)
    for key in ("learning_rate", "weight_decay", "ema_decay", "val_fraction",
                "road_crop_probability", "min_road_fraction", "decoder_dropout",
                "ocfe_initial_scale"):
        setattr(cfg, key, float(getattr(cfg, key)))
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    if cfg.dataset not in ("massachusetts", "deepglobe"):
        raise ValueError(f"Unsupported dataset: {cfg.dataset}")
    if cfg.optimizer != "adamw" or cfg.loss != "bce_dice":
        raise ValueError("This release uses AdamW and pure BCE + Dice")
    for key in ("epochs", "batch_size", "accumulation_steps", "val_interval",
                "full_checkpoint_interval", "crop_size", "decoder_dim"):
        value = getattr(cfg, key)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if cfg.crop_size % 32:
        raise ValueError("crop_size must be divisible by 32 for MiT-B3")
    if cfg.schedule_epochs is not None and cfg.schedule_epochs < cfg.epochs:
        raise ValueError("schedule_epochs must be >= epochs")
    if cfg.amp not in ("float16", "bfloat16", "float32"):
        raise ValueError("amp must be float16, bfloat16, or float32")
    if len(cfg.spans) != 3 or any(not isinstance(v, int) or v <= 0 for v in cfg.spans):
        raise ValueError("spans must contain three positive integers")
    if not 0 < cfg.val_fraction < 1 or not 0 < cfg.ema_decay < 1:
        raise ValueError("val_fraction and ema_decay must lie in (0, 1)")
    if not 0 <= cfg.road_crop_probability <= 1 or not 0 <= cfg.min_road_fraction <= 1:
        raise ValueError("Invalid crop probabilities")
    if cfg.learning_rate <= 0 or cfg.weight_decay < 0 or cfg.warmup_epochs < 0:
        raise ValueError("Invalid learning-rate schedule")
    if cfg.num_workers < 0 or cfg.val_workers < 0:
        raise ValueError("Worker counts must be non-negative")


def config_dict(cfg):
    result = asdict(cfg)
    result["spans"] = list(cfg.spans)
    return result
