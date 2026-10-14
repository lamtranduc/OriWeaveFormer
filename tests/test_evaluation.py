"""CPU tests for evaluation geometry, pooling, and validation-only tuning."""
from dataclasses import asdict
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from oriweaveformer import evaluate


class IdentityLogits(torch.nn.Module):
    def forward(self, image):
        return {"logits": image[:, :1]}


@pytest.mark.parametrize("height,width", [(3, 5), (5, 5), (1, 7)])
def test_all_d4_transforms_have_correct_inverses(height, width):
    image = torch.arange(2 * height * width).reshape(1, 2, height, width)
    for turns in range(4):
        for reflected in (False, True):
            transformed = evaluate.d4_transform(image, turns, reflected)
            assert torch.equal(evaluate.d4_inverse(transformed, turns, reflected), image)
    if min(height, width) > 1:
        views = [evaluate.d4_transform(image, k, r).flatten().tolist()
                 for k in range(4) for r in (False, True)]
        assert len({tuple(view) for view in views}) == 8


@pytest.mark.parametrize("shape,window,stride", [
    ((11, 17), 8, 3),   # Non-square, both stride padding and overlap.
    ((16, 24), 8, 4),   # Exact boundaries.
    ((3, 5), 8, 4),     # Smaller than a window.
    ((1, 1), 8, 4),     # Degenerate reflect dimension.
    ((1, 19), 8, 5),
])
@pytest.mark.parametrize("tta", [False, True])
def test_tiled_identity_covers_every_native_pixel(shape, window, stride, tta):
    generator = torch.Generator().manual_seed(7)
    image = torch.randn(1, 3, *shape, generator=generator)
    probability = evaluate.sliding_predict(IdentityLogits(), image, window, stride, tta=tta)
    assert probability.shape == (1, 1, *shape)
    torch.testing.assert_close(probability, image[:, :1].sigmoid(), atol=3e-7, rtol=1e-6)
    assert torch.isfinite(probability).all()


def test_padding_matches_historical_reflect_on_normal_images():
    image = torch.arange(1 * 3 * 11 * 17).reshape(1, 3, 11, 17).float()
    padded, original = evaluate.pad_native(image, window=8, stride=4)
    assert original == (11, 17)
    torch.testing.assert_close(padded, F.pad(image, (0, 3, 0, 1), mode="reflect"))


def test_d4_averages_probabilities_instead_of_logits():
    class CornerLogits(torch.nn.Module):
        def forward(self, image):
            return image[:, :1, :1, :1].expand(1, 1, 4, 4)

    image = torch.zeros(1, 3, 4, 4)
    image[0, 0, 0, -1], image[0, 0, -1, 0], image[0, 0, -1, -1] = 1, 2, 4
    probability = evaluate.sliding_predict(CornerLogits(), image, window=4, stride=4)
    expected = torch.tensor([0., 1., 2., 4.]).sigmoid().mean()
    torch.testing.assert_close(probability, expected.expand_as(probability))
    assert not torch.isclose(expected, torch.tensor(1.75).sigmoid())


def test_edge_tile_is_included_once_and_gaps_are_rejected():
    assert evaluate.tile_starts(17, 8, 4) == [0, 4, 8, 9]
    assert evaluate.tile_starts(16, 8, 4) == [0, 4, 8]
    with pytest.raises(ValueError, match="stride"):
        evaluate.sliding_predict(IdentityLogits(), torch.ones(1, 3, 10, 10), 4, 5)


def test_threshold_histogram_matches_exhaustive_validation_search():
    probabilities = np.array([.05, .27, .30003, .42, .61, .82, 1.0], dtype=np.float32)
    target = np.array([0, 0, 1, 1, 0, 1, 1], dtype=bool)
    bins = 10001
    positive, negative = evaluate.probability_histogram(probabilities, target, bins)
    threshold, result = evaluate.select_threshold(positive, negative)
    ids = (probabilities * (bins - 1)).astype(np.int32)
    scores = []
    for index in range(2500, 7501):
        tp, fp, fn = evaluate.confusion(ids >= index, target)
        scores.append(2 * tp / max(2 * tp + fp + fn, 1))
    expected = 2500 + int(np.argmax(scores))
    assert threshold == expected / 10000
    assert tuple(result[k] for k in ("tp", "fp", "fn")) == evaluate.confusion(ids >= expected, target)
    assert evaluate.select_threshold(np.zeros(10001, dtype=int), np.ones(10001, dtype=int))[0] == .25


def test_test_labels_cannot_change_validation_threshold_and_metrics_are_pooled(monkeypatch):
    cases = {
        "val": (np.array([.2, .4, .7]), np.array([0, 1, 1], dtype=bool)),
        "test_a": (np.array([.8]), np.array([1], dtype=bool)),
        "test_b": (np.full(9, .6), np.zeros(9, dtype=bool)),
    }
    accessed = []

    def fake_prediction(model, pair, **kwargs):
        accessed.append(pair)
        return cases[pair]

    monkeypatch.setattr(evaluate, "aligned_prediction", fake_prediction)
    first = evaluate.evaluate_pairs(None, ["val"], ["test_a", "test_b"])
    assert accessed == ["val", "test_a", "test_b"]
    fixed = first["test_metrics"]["fixed_0.5"]
    assert (fixed["tp"], fixed["fp"], fixed["fn"]) == (1, 9, 0)
    assert fixed["f1"] == pytest.approx(2 / 11, abs=1e-7)
    assert fixed["mean_image_f1"] == pytest.approx(.5, abs=1e-7)
    cases["test_b"] = (np.full(9, .1), np.ones(9, dtype=bool))
    second = evaluate.evaluate_pairs(None, ["val"], ["test_a", "test_b"])
    assert first["selected_threshold"] == second["selected_threshold"]
    assert first["validation_metrics"] == second["validation_metrics"]
    assert first["test_metrics"] != second["test_metrics"]


def test_pair_loading_uses_rgb_and_grayscale_gt_threshold(tmp_path):
    import cv2

    image_path, mask_path = tmp_path / "image.png", tmp_path / "mask.png"
    bgr = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    mask = np.array([[[127, 127, 127], [128, 128, 128]]], dtype=np.uint8)
    assert cv2.imwrite(str(image_path), bgr)
    assert cv2.imwrite(str(mask_path), mask)
    image, target = evaluate.load_native_pair((image_path, mask_path))
    np.testing.assert_array_equal(image, bgr[..., ::-1])
    np.testing.assert_array_equal(target, [[False, True]])


def test_shape_mismatch_requires_explicit_legacy_option(monkeypatch):
    monkeypatch.setattr(evaluate, "load_native_pair", lambda pair: (
        np.full((3, 5, 3), 128, dtype=np.uint8), np.ones((2, 4), dtype=bool)))
    with pytest.raises(ValueError, match="shapes differ"):
        evaluate.aligned_prediction(IdentityLogits(), ("image", "mask"), 8, 4)
    probability, target = evaluate.aligned_prediction(
        IdentityLogits(), ("image", "mask"), 8, 4, allow_shape_crop=True)
    assert probability.shape == target.shape == (2, 4)


def test_split_identity_is_portable_order_sensitive_and_overlap_rejected(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    pairs = [(first / "a.jpg", first / "a.png"), (first / "b.jpg", first / "b.png")]
    copied = [(second / "a.jpg", second / "a.png"), (second / "b.jpg", second / "b.png")]
    assert evaluate.split_identity(pairs, first) == evaluate.split_identity(copied, second)
    assert evaluate.split_identity(pairs, first)["ordered_pairs_sha256"] != evaluate.split_identity(
        pairs[::-1], first)["ordered_pairs_sha256"]
    with pytest.raises(ValueError, match="overlap"):
        evaluate.check_evaluation_splits(pairs[:1], pairs)
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate.check_evaluation_splits([pairs[0], pairs[0]], pairs[1:])


@pytest.mark.parametrize("probabilities", [np.array([float("nan")]), np.array([1.1])])
def test_invalid_probabilities_are_rejected(probabilities):
    with pytest.raises(ValueError, match="Probabilities"):
        evaluate.probability_histogram(probabilities, np.array([True]))


@pytest.mark.parametrize("container", ["cfg", "config"])
@pytest.mark.parametrize("field,bad_value", [
    ("encoder", "mit_b1"), ("spans", [4, 3, 9]), ("decoder_dim", 128),
    ("decoder_dropout", .2), ("ocfe_initial_scale", .2),
])
def test_checkpoint_model_configuration_mismatch_is_rejected(container, field, bad_value):
    from oriweaveformer.config import Config

    with pytest.raises(ValueError, match=field):
        evaluate.validate_checkpoint_config({container: {field: bad_value}}, Config())


def test_checkpoint_config_accepts_historical_aliases_and_ignores_pretrained():
    from oriweaveformer.config import Config

    evaluate.validate_checkpoint_config({"cfg": {
        "encoder": "tu-mit_b3", "spans": [4, 3, 8], "pretrained": "imagenet"}},
        Config(pretrained=None))


def test_end_to_end_cpu_evaluation_records_provenance_and_preserves_results(tmp_path, monkeypatch):
    import cv2
    from oriweaveformer.config import Config
    from oriweaveformer import data

    class TinyModel(torch.nn.Module):
        def __init__(self, encoder=None, pretrained=None, *args):
            super().__init__()
            assert pretrained is None
            self.head = torch.nn.Conv2d(3, 1, 1)

        def forward(self, image):
            return {"logits": self.head(image)}

    monkeypatch.setitem(sys.modules, "oriweaveformer.model", SimpleNamespace(
        OriWeaveFormer=TinyModel))
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    pairs = {}
    for split in ("val", "test"):
        image_path, mask_path = tmp_path / f"{split}.png", tmp_path / f"{split}_mask.png"
        assert cv2.imwrite(str(image_path), np.full((3, 5, 3), 128, dtype=np.uint8))
        assert cv2.imwrite(str(mask_path), np.full((3, 5), 255, dtype=np.uint8))
        pairs[split] = [(image_path, mask_path)]
    monkeypatch.setattr(data, "collect_pairs", lambda cfg, split: pairs[split])
    cfg = Config(data_root=str(tmp_path), pretrained=None)
    model = TinyModel()
    checkpoint, output = tmp_path / "checkpoint.pt", tmp_path / "result.json"
    torch.save({"ema": model.state_dict(), "epoch": 2, "config": asdict(cfg)}, checkpoint)
    result = evaluate.run_evaluation(cfg, checkpoint, output, window=4, stride=2, device="cpu")
    stored = json.loads(output.read_text(encoding="utf-8"))
    assert stored["checkpoint_sha256"] == evaluate.sha256_file(checkpoint)
    assert stored["checkpoint_weights"] == "ema"
    assert stored["checkpoint_epoch"] == 2
    assert stored["runtime"]["amp_actual"] == "float32"
    assert stored["protocol"]["scales"] == [1.0]
    assert stored["protocol"]["threshold_selection"]["split"] == "validation"
    assert stored["splits"]["validation"]["pairs"] == [["val.png", "val_mask.png"]]
    assert result["test_metrics"] == stored["test_metrics"]
    before = output.read_bytes()
    with pytest.raises(FileExistsError, match="preserving"):
        evaluate.run_evaluation(cfg, checkpoint, output, window=4, stride=2, device="cpu")
    assert output.read_bytes() == before


def _distributed_fake_prediction(model, pair, **kwargs):
    cases = {
        "v0": ([.2, .4, .7], [0, 1, 1]),
        "t0": ([.8], [1]),
        "t1": ([.1, .6, .8], [0, 0, 1]),
        "t2": ([.3, .1], [1, 1]),
    }
    probability, target = cases[pair]
    return np.array(probability), np.array(target, dtype=bool)


def _distributed_worker(rank, rendezvous_uri, output_directory):
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo", init_method=rendezvous_uri, rank=rank, world_size=2,
        timeout=timedelta(seconds=45))
    try:
        evaluate.aligned_prediction = _distributed_fake_prediction
        result = evaluate.evaluate_pairs(
            None, ["v0"], ["t0", "t1", "t2"], rank=rank, world_size=2)
        (Path(output_directory) / f"rank_{rank}.json").write_text(json.dumps(result), encoding="utf-8")
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(
    os.name == "nt" or not torch.distributed.is_gloo_available(),
    reason="file rendezvous is unsupported by this Windows Gloo build",
)
def test_two_process_image_sharding_matches_single_process(tmp_path, monkeypatch):
    # Rank one owns no validation images: every rank must still join every
    # reduction, and both must use the same validation-selected threshold.
    torch.multiprocessing.spawn(
        _distributed_worker, args=((tmp_path / "rendezvous").as_uri(), str(tmp_path)),
        nprocs=2, join=True)
    monkeypatch.setattr(evaluate, "aligned_prediction", _distributed_fake_prediction)
    expected = evaluate.evaluate_pairs(None, ["v0"], ["t0", "t1", "t2"])
    for rank in range(2):
        result = json.loads((tmp_path / f"rank_{rank}.json").read_text(encoding="utf-8"))
        assert result["selected_threshold"] == expected["selected_threshold"]
        assert result["validation_metrics"] == expected["validation_metrics"]
        for protocol in ("fixed_0.5", "val_tuned"):
            assert result["test_metrics"][protocol] == pytest.approx(expected["test_metrics"][protocol])
