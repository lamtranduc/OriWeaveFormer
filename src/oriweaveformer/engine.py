"""AdamW + EMA training with run snapshots and single-rank epoch logging."""
import importlib.metadata
import json
import math
import os
import platform
import random
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .checkpoint import atomic_json, atomic_save, extract_weights, sha256_file
from .config import config_dict, validate_config
from .data import RoadDataset
from .ema import EMA
from .losses import bce_dice_loss
from .metrics import pooled_metrics
from .model import OriWeaveFormer
from .prepare_data import build_manifest


def _schedule(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.5 * (1 + math.cos(math.pi * progress))


def _seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)
    import cv2
    cv2.setNumThreads(0)
    info = torch.utils.data.get_worker_info()
    if hasattr(info.dataset.transform, "set_random_seed"):
        info.dataset.transform.set_random_seed(seed)


def _snapshot(cfg, manifest, world, amp, device):
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.yaml"
    resolved = config_dict(cfg)
    if config_path.exists():
        old = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        ignored = {"resume", "data_root", "output_dir", "init_checkpoint", "split_manifest"}
        differences = [k for k in resolved if k not in ignored and old.get(k) != resolved[k]]
        if differences:
            raise ValueError(f"Existing run configuration differs: {differences}; choose a new output_dir")
        if not cfg.resume:
            raise FileExistsError("Existing run; use --resume or a new output_dir")
    else:
        with config_path.open("x", encoding="utf-8") as handle:
            yaml.safe_dump(resolved, handle, sort_keys=False)
    manifest_path = output / "splits.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["sha256"] != manifest["sha256"]:
            raise ValueError("Dataset split changed since this run was created")
    else:
        atomic_json(manifest, manifest_path)
    environment = {
        "python": platform.python_version(), "platform": platform.platform(),
        "world_size": world, "amp_requested": cfg.amp, "amp_actual": amp,
        "gpu": torch.cuda.get_device_name(device), "cuda": torch.version.cuda,
        "batch_per_process": cfg.batch_size,
        "effective_batch": cfg.batch_size * world * cfg.accumulation_steps,
        "packages": {name: importlib.metadata.version(name) for name in (
            "torch", "torchvision", "segmentation-models-pytorch", "timm",
            "numpy", "albumentations", "opencv-python-headless", "PyYAML")},
        "source_sha256": {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "resume": cfg.resume,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    atomic_json({"config": resolved, "environment": environment}, output / f"invocations/{stamp}.json")
    if not (output / "environment.json").exists():
        atomic_json(environment, output / "environment.json")


def _load_optimizer_state(saved, singular, plural):
    if singular in saved:
        return saved[singular]
    values = saved.get(plural, [])
    if len(values) != 1:
        raise ValueError(f"Checkpoint has no single {singular}; use weight initialization instead")
    return values[0]


def run_training(cfg):
    validate_config(cfg)
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a CUDA GPU")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank, local_rank = int(os.environ.get("RANK", "0")), int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    distributed = world > 1
    if distributed:
        dist.init_process_group("nccl")
    try:
        _train(cfg, world, rank, device, distributed)
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


def _train(cfg, world, rank, device, distributed):
    def barrier():
        if distributed:
            dist.barrier()

    def require_finite(value, message):
        finite = torch.tensor(int(torch.isfinite(value).all()), device=device)
        if distributed:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            raise FloatingPointError(message + "; last saved checkpoint remains available")

    seed = cfg.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    amp_name = cfg.amp
    if amp_name == "bfloat16" and not torch.cuda.is_bf16_supported():
        amp_name = "float16"
    amp_dtype = getattr(torch, amp_name)
    train_set = RoadDataset(cfg, "train")
    if hasattr(train_set.transform, "set_random_seed"):
        train_set.transform.set_random_seed(seed)
    sampler = DistributedSampler(train_set, num_replicas=world, rank=rank,
                                 seed=cfg.seed, shuffle=True, drop_last=True) if distributed else None
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=sampler is None, sampler=sampler,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
        worker_init_fn=_seed_worker, generator=generator)
    if len(train_loader) == 0:
        raise ValueError("Training split is smaller than one complete global batch")
    output = Path(cfg.output_dir)
    if output.resolve().is_relative_to(Path(__file__).resolve().parents[2] / "experiments"):
        raise ValueError("Use runs/ for execution; experiments/ contains frozen paper records")
    last_path = output / "last.pt"
    continuing = cfg.resume and last_path.exists()
    if cfg.resume and not continuing:
        raise FileNotFoundError("--resume requires last.pt; for weights only use --init-checkpoint")
    if rank == 0:
        manifest = build_manifest(cfg, require_test=False)
        _snapshot(cfg, manifest, world, amp_name, device)
    barrier()
    # Only rank zero fetches ImageNet weights; DDP broadcasts the initialized model.
    pretrained = cfg.pretrained if rank == 0 and not continuing and not cfg.init_checkpoint else None
    model = OriWeaveFormer(
        cfg.encoder, pretrained, cfg.spans, cfg.decoder_dim,
        cfg.decoder_dropout, cfg.ocfe_initial_scale).to(device)
    if cfg.init_checkpoint and not continuing:
        saved = torch.load(cfg.init_checkpoint, map_location="cpu", weights_only=False)
        state, _ = extract_weights(saved)
        model.load_state_dict(state, strict=True)
        del saved, state
    wrapped = DistributedDataParallel(model, device_ids=[device.index], broadcast_buffers=False) \
        if distributed else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / cfg.accumulation_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: _schedule(
        step, cfg.warmup_epochs * updates_per_epoch,
        (cfg.schedule_epochs or cfg.epochs) * updates_per_epoch))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_name == "float16")
    ema = EMA(model, cfg.ema_decay) if rank == 0 else None
    start_epoch, global_step, best_f1 = 1, 0, 0.0
    if continuing:
        # Serialize large checkpoint disk reads across processes.
        for owner in range(world):
            if owner == rank:
                saved = torch.load(last_path, map_location="cpu", weights_only=False)
                model.load_state_dict(saved["model"], strict=True)
                optimizer.load_state_dict(_load_optimizer_state(saved, "optimizer", "optimizers"))
                scheduler.load_state_dict(_load_optimizer_state(saved, "scheduler", "schedulers"))
                scaler.load_state_dict(saved["scaler"])
                start_epoch, global_step = saved["epoch"] + 1, saved.get("global_step", saved.get("step", 0))
                best_f1 = saved.get("best_f1", 0.0)
                if ema is not None:
                    ema.shadow.load_state_dict(saved["ema"], strict=True)
                    ema.updates = saved.get("ema_updates", 0)
                states = saved.get("random_states", [])
                if len(states) == world:
                    state = states[rank]
                    random.setstate(state["python"])
                    np.random.set_state(state["numpy"])
                    torch.set_rng_state(state["torch"])
                    torch.cuda.set_rng_state(state["cuda"], device)
                    generator.set_state(state["loader"])
                del saved
            barrier()
    val_loader = None
    history_path = output / "history.json"
    history = []
    if rank == 0:
        val_set = RoadDataset(cfg, "val")
        val_loader = DataLoader(val_set, batch_size=1, num_workers=cfg.val_workers,
                                pin_memory=True, worker_init_fn=_seed_worker)
        if continuing and history_path.exists():
            history = [row for row in json.loads(history_path.read_text()) if row["epoch"] < start_epoch]
        print(f"OriWeaveFormer world={world} train={len(train_set)} val={len(val_set)} "
              f"batch_per_process={cfg.batch_size} accum={cfg.accumulation_steps} amp={amp_name}", flush=True)
    for epoch in range(start_epoch, cfg.epochs + 1):
        if sampler:
            sampler.set_epoch(epoch)
        wrapped.train()
        optimizer.zero_grad(set_to_none=True)
        started, running, batches = time.perf_counter(), 0.0, 0
        for index, (image, target) in enumerate(train_loader):
            image, target = image.to(device, non_blocking=True), target.to(device, non_blocking=True)
            update = (index + 1) % cfg.accumulation_steps == 0 or index + 1 == len(train_loader)
            group_start = (index // cfg.accumulation_steps) * cfg.accumulation_steps
            group_size = min(cfg.accumulation_steps, len(train_loader) - group_start)
            sync = wrapped.no_sync() if distributed and not update else nullcontext()
            with sync:
                with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_name != "float32"):
                    loss = bce_dice_loss(wrapped(image)["logits"], target)
                require_finite(loss, f"Non-finite loss at epoch {epoch}, batch {index + 1}")
                scaler.scale(loss / group_size).backward()
            if update:
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                require_finite(norm, f"Non-finite gradient at epoch {epoch}, batch {index + 1}")
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                if ema:
                    ema.update(model)
                optimizer.zero_grad(set_to_none=True)
            running += loss.item()
            batches += 1
            global_step += 1
        totals = torch.tensor([running, batches], dtype=torch.float64, device=device)
        if distributed:
            dist.all_reduce(totals)
        torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - started
        save_full = epoch % cfg.full_checkpoint_interval == 0 or epoch == cfg.epochs
        random_states = None
        if save_full:
            state = {"python": random.getstate(), "numpy": np.random.get_state(),
                     "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device),
                     "loader": generator.get_state()}
            random_states = [None] * world
            if distributed:
                dist.all_gather_object(random_states, state)
            else:
                random_states[0] = state
        if rank == 0:
            metrics, improved = None, False
            if epoch % cfg.val_interval == 0 or epoch == cfg.epochs:
                metrics = pooled_metrics(ema.shadow, val_loader, device, amp_dtype)
                if metrics["f1"] > best_f1:
                    best_f1, improved = metrics["f1"], True
                    atomic_save({"ema": ema.shadow.state_dict(), "epoch": epoch,
                                 "metrics": metrics, "config": config_dict(cfg)}, output / "best.pt")
            atomic_save({"ema": ema.shadow.state_dict(), "epoch": epoch,
                         "best_f1": best_f1, "config": config_dict(cfg)}, output / "latest_weights.pt")
            if save_full:
                atomic_save({"model": model.state_dict(), "ema": ema.shadow.state_dict(),
                             "ema_updates": ema.updates, "optimizer": optimizer.state_dict(),
                             "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                             "epoch": epoch, "global_step": global_step, "best_f1": best_f1,
                             "config": config_dict(cfg), "random_states": random_states}, last_path)
            row = {"epoch": epoch, "loss": float(totals[0] / totals[1]),
                   "train_seconds": train_seconds, "seconds": time.perf_counter() - started,
                   "learning_rate": optimizer.param_groups[0]["lr"], "metrics": metrics,
                   "best_f1": best_f1, "skipped": 0}
            history.append(row)
            atomic_json(history, history_path)
            line = f"epoch {epoch}: loss={row['loss']:.4f} ({row['seconds']:.0f}s)"
            if metrics:
                line += " " + " ".join(f"{label}={metrics[key]:.4f}" for label, key in (
                    ("P", "precision"), ("R", "recall"), ("F1", "f1"), ("IoU", "iou")))
            print(line + (" *best" if improved else ""), flush=True)
        barrier()
