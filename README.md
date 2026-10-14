# OriWeaveFormer

A standalone research release of the final OriWeaveFormer road segmentation model for Massachusetts Roads and DeepGlobe.

## Layout

```text
configs/
  datasets/                     Dataset fragments
  models/                       Final OriWeaveFormer model settings
  experiments/                  Massachusetts and DeepGlobe experiment settings
data/
  massachusetts/                Dataset documentation and split manifest
  deepglobe/                    Dataset documentation and split manifest
src/oriweaveformer/             Importable model, training, and evaluation code
experiments/oriweaveformer_final/
  massachusetts/                Dataset-specific evidence
  deepglobe/                    Dataset-specific evidence and metrics
  summary/                      Cross-dataset result exports
docs/                           Architecture, walkthrough, and evaluation protocol
runs/                           New local outputs, outside historical evidence
checkpoints/                    Final EMA checkpoints for both datasets
```

See [architecture summary](docs/architecture.md), [complete Vietnamese layer specification](docs/architecture-complete-vi.md), and [architecture walkthrough (vi)](docs/oriweaveformer_walkthrough_vi.md).

## Install

From the release directory, use a virtual environment and install the package:

```bash
python -m venv .venv
# Activate .venv using your shell, then:
pip install -e .
```

The included `uv.lock` records a dependency resolution prepared for this release. With `uv` installed, `uv sync --frozen` uses that resolution. Select a PyTorch installation compatible with the target GPU and driver when preparing a training machine. Training requires CUDA; the two final EMA checkpoints are bundled, while datasets and GPU resources are not.

## Register data

Prepare locally acquired, paired image and mask files as described in [Massachusetts](data/massachusetts/README.md) or [DeepGlobe](data/deepglobe/README.md), then record the discovered split:

```bash
oriweaveformer-prepare-data --config configs/experiments/oriweaveformer_massachusetts.yaml --data-root /path/to/massachusetts --output data/massachusetts/splits.json
oriweaveformer-prepare-data --config configs/experiments/oriweaveformer_deepglobe.yaml --data-root /path/to/deepglobe --output data/deepglobe/splits.json
```

Pass the saved manifest through `--split-manifest` during training or evaluation to fix membership explicitly. Data preparation records pairs; it does not download data or create a new dataset license.

## Train

Configuration files compose dataset and model settings through `load_experiment`. Command-line overrides specify machine-local paths.

```bash
oriweaveformer-train --config configs/experiments/oriweaveformer_massachusetts.yaml --data-root /path/to/massachusetts --split-manifest data/massachusetts/splits.json
oriweaveformer-train --config configs/experiments/oriweaveformer_deepglobe.yaml --data-root /path/to/deepglobe --split-manifest data/deepglobe/splits.json
```

As detailed in Section 4.2 of the paper, training is executed on **512×512 crops** with an effective batch size of 8 (250 epochs on Massachusetts and 150 epochs on DeepGlobe), optimized via AdamW ($\eta = 2 \times 10^{-4}$, weight decay $1 \times 10^{-4}$, 3-epoch warmup with cosine annealing schedule) and balanced road-aware patch sampling ($p=0.7$).

`batch_size` is per process. For Massachusetts on 2 GPUs (4 images per GPU), run:

```bash
torchrun --standalone --nproc_per_node=2 -m oriweaveformer.train --config configs/experiments/oriweaveformer_massachusetts.yaml --data-root /path/to/massachusetts --split-manifest data/massachusetts/splits.json
```

For DeepGlobe on a single GPU with batch size 8:

```bash
oriweaveformer-train --config configs/experiments/oriweaveformer_deepglobe.yaml --data-root /path/to/deepglobe --split-manifest data/deepglobe/splits.json
```

Effective optimizer batch size is `batch_size × world_size × accumulation_steps = 8`. Use fresh output directories for new runs. Training validation uses deterministic center crops and a fixed probability threshold of 0.5 for checkpoint selection.

## Evaluate and export

```bash
oriweaveformer-evaluate --config configs/experiments/oriweaveformer_massachusetts.yaml --data-root /path/to/massachusetts --split-manifest data/massachusetts/splits.json --ckpt checkpoints/massachusetts.pt --out runs/eval/massachusetts_metrics.json
oriweaveformer-evaluate --config configs/experiments/oriweaveformer_deepglobe.yaml --data-root /path/to/deepglobe --split-manifest data/deepglobe/splits.json --ckpt checkpoints/deepglobe.pt --out runs/eval/deepglobe_metrics.json
oriweaveformer-export-results --experiment experiments/oriweaveformer_final
```

Final evaluation retains native image resolution ($1024 \times 1024$), uses 1024-pixel sliding windows with stride 512 and D4 eight-way test-time augmentation (8-TTA), and selects the threshold using validation data only. The selected threshold is then applied to the held-out test split, reproducing the SOTA benchmarks reported in Table 1 of the paper. Optional `--device`, `--amp`, and `--weights` arguments select execution and checkpoint settings; consult `oriweaveformer-evaluate --help` for accepted values. See the [evaluation protocol](docs/evaluation.md).

Historical evidence is stored under the relevant dataset in `experiments/oriweaveformer_final/`. The export command generates `experiments/oriweaveformer_final/summary/results.csv` from that evidence.

## Reproducibility and checkpoints

Model state-dict names remain compatible with the supplied OriWeaveFormer architecture. The bundled `checkpoints/massachusetts.pt` (trained 250 epochs on 512×512 crops) and `checkpoints/deepglobe.pt` (trained 150 epochs on 512×512 crops) contain the final EMA weights and load with `strict=True`, matching the published paper specifications and benchmarks. Resume restores the saved training state, but bitwise continuation is not guaranteed because worker random state and CUDA execution can differ.
