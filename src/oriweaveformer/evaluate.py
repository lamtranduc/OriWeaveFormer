"""Native-resolution, tile-local D4 evaluation of one OriWeaveFormer checkpoint.

This preserves the recorded OriWeaveFormer evaluation protocol: 1024/512 overlapping
windows, eight D4 views, probability averaging, and a threshold selected on
validation pixels only. It supports one CPU/GPU or image-sharded ``torchrun``.
No image resizing, multiscale inference, or checkpoint averaging is performed.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import cv2
import numpy as np
import torch
import torch.distributed as dist


MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)


def d4_transform(x: torch.Tensor, turns: int, reflected: bool = False):
    """Reflect horizontally first, then rotate counterclockwise by 90*turns."""
    if reflected:
        x = x.flip(-1)
    return torch.rot90(x, turns, dims=(-2, -1))


def d4_inverse(x: torch.Tensor, turns: int, reflected: bool = False):
    x = torch.rot90(x, -turns, dims=(-2, -1))
    return x.flip(-1) if reflected else x


def tile_starts(length: int, window: int, stride: int) -> list[int]:
    """Include the final edge tile even when the stride does not land on it."""
    if not 0 < stride <= window or length < window:
        raise ValueError("Require 0 < stride <= window <= padded dimension")
    starts = list(range(0, length - window + 1, stride))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


def _reflect_indices(length: int, target: int, device) -> torch.Tensor:
    # Equivalent to reflect padding for ordinary inputs, and defined even
    # when the requested padding exceeds the input or a dimension is one.
    indices = torch.arange(target, device=device)
    if length == 1:
        return torch.zeros_like(indices)
    period = 2 * length - 2
    indices %= period
    return torch.minimum(indices, period - indices)


def pad_native(x: torch.Tensor, window: int, stride: int):
    """Pad bottom/right to stride multiples and at least one full window."""
    if x.ndim != 4 or x.shape[0] != 1 or min(x.shape[-2:]) < 1:
        raise ValueError("Expected one nonempty image with shape 1,C,H,W")
    if not 0 < stride <= window:
        raise ValueError("Require 0 < stride <= window")
    height, width = x.shape[-2:]
    padded_h = ((max(height, window) + stride - 1) // stride) * stride
    padded_w = ((max(width, window) + stride - 1) // stride) * stride
    if padded_h != height:
        x = x.index_select(-2, _reflect_indices(height, padded_h, x.device))
    if padded_w != width:
        x = x.index_select(-1, _reflect_indices(width, padded_w, x.device))
    return x, (height, width)


@torch.inference_mode()
def sliding_predict(model, x: torch.Tensor, window: int = 1024,
                    stride: int = 512, *, tta: bool = True,
                    amp_dtype: torch.dtype | None = None) -> torch.Tensor:
    """Return native-sized probabilities, averaging views and overlaps equally.

    ``x`` is an ImageNet-normalized 1,C,H,W tensor on the model's device.
    The model must return full-window binary logits (or a dict with logits).
    Padding is removed after the final average, so border pixels are scored.
    """
    x, (native_h, native_w) = pad_native(x, window, stride)
    height, width = x.shape[-2:]
    acc = torch.zeros((1, 1, height, width), device=x.device, dtype=torch.float32)
    counts = torch.zeros_like(acc)
    ys, xs = tile_starts(height, window, stride), tile_starts(width, window, stride)
    transforms = [(k, reflected) for k in range(4) for reflected in (False, True)]
    if not tta:
        transforms = [(0, False)]
    # Keep historical accumulation order: transform, row, column.
    for turns, reflected in transforms:
        for y in ys:
            for left in xs:
                tile = d4_transform(x[..., y:y + window, left:left + window],
                                    turns, reflected)
                context = (torch.autocast(x.device.type, dtype=amp_dtype)
                           if amp_dtype is not None else nullcontext())
                with context:
                    output = model(tile)
                    logits = output["logits"] if isinstance(output, dict) else output
                    if logits.shape != (1, 1, window, window):
                        raise ValueError("Model must return 1,1,window,window logits; "
                                         f"received {tuple(logits.shape)}")
                    probability = torch.sigmoid(logits)
                probability = d4_inverse(probability.float(), turns, reflected)
                if not torch.isfinite(probability).all():
                    raise ValueError("Model returned non-finite probabilities")
                acc[..., y:y + window, left:left + window] += probability
                counts[..., y:y + window, left:left + window] += 1
    return (acc / counts)[..., :native_h, :native_w]


def load_native_pair(pair):
    image_path, mask_path = map(Path, pair)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot decode image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if raw_mask is None:
        raise FileNotFoundError(f"Cannot decode mask: {mask_path}")
    if raw_mask.ndim == 3:
        raw_mask = cv2.cvtColor(raw_mask, cv2.COLOR_BGR2GRAY)
    if raw_mask.ndim != 2:
        raise ValueError(f"Expected a two-dimensional mask: {mask_path}")
    return image, raw_mask > 127


def aligned_prediction(model, pair, window=1024, stride=512, *,
                       device="cpu", amp_dtype=None, allow_shape_crop=False):
    image, target = load_native_pair(pair)
    if image.shape[:2] != target.shape and not allow_shape_crop:
        raise ValueError(f"Image/mask shapes differ for {pair[0]}: "
                         f"{image.shape[:2]} != {target.shape}; check the dataset "
                         "or explicitly enable historical top-left shape cropping")
    normalized = (image.astype(np.float32) / 255.0 - MEAN) / STD
    x = torch.from_numpy(normalized.transpose(2, 0, 1).copy())[None].to(device)
    probability = sliding_predict(model, x, window, stride,
                                  amp_dtype=amp_dtype)[0, 0].cpu().numpy()
    if probability.shape != target.shape:
        height = min(probability.shape[0], target.shape[0])
        width = min(probability.shape[1], target.shape[1])
        probability, target = probability[:height, :width], target[:height, :width]
    return probability, target


def confusion(prediction, target) -> tuple[int, int, int]:
    prediction, target = np.asarray(prediction, bool), np.asarray(target, bool)
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes must match")
    return (int(np.logical_and(prediction, target).sum()),
            int(np.logical_and(prediction, ~target).sum()),
            int(np.logical_and(~prediction, target).sum()))


def metrics(tp: int, fp: int, fn: int) -> dict:
    """Historical epsilon convention; pooled counts are retained for audit."""
    eps = 1e-7
    precision, recall = tp / (tp + fp + eps), tp / (tp + fn + eps)
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall + eps),
            "iou": tp / (tp + fp + fn + eps), "tp": tp, "fp": fp, "fn": fn}


def probability_histogram(probability, target, bins: int = 10001):
    if bins < 3:
        raise ValueError("At least three histogram bins are required")
    probability, target = np.asarray(probability), np.asarray(target, dtype=bool)
    if probability.shape != target.shape:
        raise ValueError("Probability and target shapes must match")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Probabilities must be finite and between zero and one")
    ids = np.clip((probability * (bins - 1)).astype(np.int32), 0, bins - 1)
    return (np.bincount(ids[target], minlength=bins),
            np.bincount(ids[~target], minlength=bins))


def select_threshold(positive, negative, bins: int | None = None):
    """Select pooled validation F1 maximum in [0.25,0.75], ties going lower."""
    positive, negative = np.asarray(positive), np.asarray(negative)
    bins = len(positive) if bins is None else bins
    if bins < 3 or positive.shape != (bins,) or negative.shape != (bins,):
        raise ValueError("Expected equal one-dimensional positive/negative histograms")
    if np.any(positive < 0) or np.any(negative < 0) or positive.sum() + negative.sum() == 0:
        raise ValueError("Threshold selection requires nonempty nonnegative histograms")
    tp, fp = np.cumsum(positive[::-1])[::-1], np.cumsum(negative[::-1])[::-1]
    fn = positive.sum() - tp
    denominator = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=np.float64),
                   where=denominator > 0)
    lower, upper = round(.25 * (bins - 1)), round(.75 * (bins - 1))
    index = lower + int(np.argmax(f1[lower:upper + 1]))
    return index / (bins - 1), metrics(int(tp[index]), int(fp[index]), int(fn[index]))


def _reduce_array(array, device):
    if dist.is_available() and dist.is_initialized():
        tensor = torch.from_numpy(array).to(device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.cpu().numpy()
    return array


def evaluate_pairs(model, validation_pairs, test_pairs, *, window=1024,
                   stride=512, bins=10001, device="cpu", amp_dtype=None,
                   allow_shape_crop=False, rank=0, world_size=1):
    """Finish threshold selection on validation before reading test predictions."""
    if not validation_pairs or not test_pairs:
        raise ValueError("Both validation and test splits must be nonempty")
    positive, negative = np.zeros(bins, dtype=np.int64), np.zeros(bins, dtype=np.int64)
    prediction_options = dict(window=window, stride=stride, device=device,
                              amp_dtype=amp_dtype, allow_shape_crop=allow_shape_crop)
    for index in range(rank, len(validation_pairs), world_size):
        probability, target = aligned_prediction(model, validation_pairs[index],
                                                  **prediction_options)
        current_positive, current_negative = probability_histogram(probability, target, bins)
        positive += current_positive
        negative += current_negative
        print(f"[rank {rank}] validation image {index + 1}/{len(validation_pairs)}", flush=True)
    positive, negative = _reduce_array(positive, device), _reduce_array(negative, device)
    threshold, validation_metrics = select_threshold(positive, negative, bins)
    # Test pixels never enter a histogram or a threshold selection operation.
    thresholds = {"fixed_0.5": 0.5, "val_tuned": threshold}
    counts, image_f1 = np.zeros((2, 3), dtype=np.int64), np.zeros(2, dtype=np.float64)
    for index in range(rank, len(test_pairs), world_size):
        probability, target = aligned_prediction(model, test_pairs[index], **prediction_options)
        for row, value in enumerate(thresholds.values()):
            current = confusion(probability >= value, target)
            counts[row] += current
            image_f1[row] += metrics(*current)["f1"]
        print(f"[rank {rank}] test image {index + 1}/{len(test_pairs)}", flush=True)
    counts, image_f1 = _reduce_array(counts, device), _reduce_array(image_f1, device)
    test_metrics = {}
    for row, (name, value) in enumerate(thresholds.items()):
        test_metrics[name] = metrics(*(int(n) for n in counts[row]))
        test_metrics[name].update(threshold=value,
                                  mean_image_f1=float(image_f1[row] / len(test_pairs)))
    return {"selected_threshold": threshold, "validation_metrics": validation_metrics,
            "test_metrics": test_metrics}


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_identity(pairs, root) -> dict:
    """A portable hash of the ordered, root-relative image/mask identities."""
    root = Path(root).resolve()
    entries = []
    for pair in pairs:
        entry = []
        for path in pair:
            resolved = Path(path).resolve()
            try:
                entry.append(resolved.relative_to(root).as_posix())
            except ValueError:
                entry.append(resolved.as_posix())
        entries.append(entry)
    serialized = json.dumps(entries, ensure_ascii=False, separators=(",", ":"))
    return {"count": len(entries), "ordered_pairs_sha256":
            hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
            "hash_scope": "ordered image/mask path identities, not file contents",
            "pairs": entries}


def check_evaluation_splits(validation_pairs, test_pairs):
    for label, pairs in (("validation", validation_pairs), ("test", test_pairs)):
        if not pairs:
            raise ValueError(f"Empty {label} split")
        for column in (0, 1):
            paths = [Path(pair[column]).resolve() for pair in pairs]
            if len(set(paths)) != len(paths):
                raise ValueError(f"Duplicate {'images' if column == 0 else 'masks'} in {label}")
    validation_paths = {Path(path).resolve() for pair in validation_pairs for path in pair}
    test_paths = {Path(path).resolve() for pair in test_pairs for path in pair}
    if validation_paths & test_paths:
        raise ValueError("Validation and test splits overlap")


def validate_checkpoint_config(saved, cfg):
    """Reject architecture changes that strict state-dict loading cannot see."""
    if not isinstance(saved, dict):
        return
    fields = ("encoder", "spans", "decoder_dim", "decoder_dropout", "ocfe_initial_scale")
    for container in ("cfg", "config"):
        recorded = saved.get(container)
        if not isinstance(recorded, dict):
            continue
        for field in fields:
            if field not in recorded:
                continue
            expected, actual = recorded[field], getattr(cfg, field)
            if field == "spans":
                expected, actual = tuple(expected), tuple(actual)
            elif field == "encoder":
                # The historical roadx factory removed this timm prefix.
                expected = str(expected).removeprefix("tu-")
            elif field in ("decoder_dropout", "ocfe_initial_scale"):
                expected, actual = float(expected), float(actual)
            elif field == "decoder_dim":
                expected, actual = int(expected), int(actual)
            if expected != actual:
                raise ValueError(f"Checkpoint {container}.{field}={expected!r} conflicts "
                                 f"with evaluation config {actual!r}")


def _runtime(device_name: str, amp: str):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device_name == "auto":
        device_name = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if world_size > 1 and device.type == "cuda":
        device = torch.device("cuda", local_rank)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    owned_process_group = world_size > 1 and not dist.is_initialized()
    if owned_process_group:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    actual_amp = "float32" if device.type == "cpu" else amp
    if actual_amp == "bfloat16" and not torch.cuda.is_bf16_supported():
        actual_amp = "float16"
    amp_dtype = None if actual_amp == "float32" else getattr(torch, actual_amp)
    return device, rank, world_size, amp_dtype, actual_amp, owned_process_group


def run_evaluation(cfg, checkpoint_path, output_path, *, window=1024,
                   stride=512, bins=10001, device="auto", amp="bfloat16",
                   prefer_ema=True, allow_shape_crop=False):
    from .checkpoint import extract_weights
    from .data import collect_pairs
    from .model import OriWeaveFormer

    if not 0 < stride <= window or bins < 3:
        raise ValueError("Require 0 < stride <= window and bins >= 3")
    if amp not in ("bfloat16", "float16", "float32"):
        raise ValueError("amp must be bfloat16, float16, or float32")
    checkpoint_path, output_path = Path(checkpoint_path).resolve(), Path(output_path).resolve()
    if checkpoint_path == output_path:
        raise ValueError("Evaluation output must not overwrite the checkpoint")
    if output_path.exists():
        raise FileExistsError(f"Choose a new output path; preserving existing result: {output_path}")
    validation_pairs, test_pairs = collect_pairs(cfg, "val"), collect_pairs(cfg, "test")
    check_evaluation_splits(validation_pairs, test_pairs)
    device, rank, world_size, amp_dtype, actual_amp, owned_group = _runtime(device, amp)
    started = time.monotonic()
    try:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        validate_checkpoint_config(saved, cfg)
        state, state_source = extract_weights(saved, prefer_ema=prefer_ema)
        # All learned parameters come from the supplied checkpoint: no network
        # pretrained-weight initialization is needed during evaluation.
        model = OriWeaveFormer(
            cfg.encoder, None, cfg.spans, cfg.decoder_dim,
            cfg.decoder_dropout, cfg.ocfe_initial_scale)
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        scores = evaluate_pairs(
            model, validation_pairs, test_pairs, window=window, stride=stride,
            bins=bins, device=device, amp_dtype=amp_dtype,
            allow_shape_crop=allow_shape_crop, rank=rank, world_size=world_size)
        if rank != 0:
            return None
        result = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_epoch": saved.get("epoch") if isinstance(saved, dict) else None,
            "checkpoint_weights": state_source,
            "config": asdict(cfg),
            "protocol": {
                "name": "native_sliding_d4_8tta",
                "native_resolution": True, "scales": [1.0],
                "window": window, "stride": stride,
                "tta": "D4 8-way (tile-local)",
                "aggregation": "uniform probability average over D4 views and overlapping tiles",
                "padding": "bottom/right reflect to stride multiple and at least window; singleton dimensions replicate",
                "normalization": {"color": "RGB", "input_divisor": 255,
                                  "mean": MEAN.tolist(), "std": STD.tolist()},
                "mask": "OpenCV unchanged; multichannel BGR-to-gray; value > 127",
                "shape_policy": "top-left intersection" if allow_shape_crop else "require identical image/mask shapes",
                "threshold_selection": {
                    "split": "validation", "objective": "pooled pixel F1",
                    "bins": bins, "quantization": "floor(probability * (bins - 1))",
                    "range": [0.25, 0.75], "tie_break": "lowest threshold",
                    "test_thresholds": ["fixed 0.5", "frozen validation selection"]},
                "metric_aggregation": "pooled TP/FP/FN; mean_image_f1 is separately labelled",
                "metric_epsilon": 1e-7,
            },
            "splits": {"validation": split_identity(validation_pairs, cfg.data_root),
                       "test": split_identity(test_pairs, cfg.data_root)},
            "runtime": {"device": str(device), "world_size": world_size,
                        "amp_requested": amp, "amp_actual": actual_amp,
                        "torch": torch.__version__, "numpy": np.__version__,
                        "opencv": cv2.__version__, "python": platform.python_version(),
                        "elapsed_seconds": time.monotonic() - started},
            **scores,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation makes preservation effective even if another run
        # claims this filename after the initial check.
        with output_path.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
        return result
    finally:
        if owned_group:
            dist.destroy_process_group()


def main(argv=None):
    from .config import load_experiment

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Experiment YAML or JSON")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True, help="New JSON output path; existing files are preserved")
    parser.add_argument("--data-root")
    parser.add_argument("--split-manifest")
    parser.add_argument("--window", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--bins", type=int, default=10001)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda[:index]")
    parser.add_argument("--amp", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema",
                        help="ema prefers EMA then model; model requires raw model weights; source is recorded")
    parser.add_argument("--allow-shape-crop", action="store_true",
                        help="Explicitly reproduce historical top-left image/mask intersection")
    args = parser.parse_args(argv)
    cfg = load_experiment(args.config, overrides={
        "data_root": args.data_root,
        "split_manifest": args.split_manifest,
    })
    result = run_evaluation(
        cfg, args.ckpt, args.out, window=args.window, stride=args.stride,
        bins=args.bins, device=args.device, amp=args.amp,
        prefer_ema=args.weights == "ema", allow_shape_crop=args.allow_shape_crop)
    if result is not None:
        print(json.dumps({"output": str(Path(args.out).resolve()),
                          "selected_threshold": result["selected_threshold"],
                          "test_metrics": result["test_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
