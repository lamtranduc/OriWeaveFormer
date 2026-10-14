# Massachusetts Roads

This release consumes an already acquired local copy of paired road imagery and masks. It does not include dataset payloads, a downloader, or a redistribution grant. Record the authoritative acquisition source, dataset version or snapshot, applicable license, and preferred citation for the actual files before publication; these details were not established by the supplied OriWeaveFormer source.

## Expected layout

Point `--data-root` at a directory containing:

```text
massachusetts/
  training/
    images/
    masks/
  eval/
    images/
    masks/
```

Discovery can find the named directories beneath the data root. Use an unambiguous root containing one intended dataset copy. Images are paired by stem; an optional image suffix `_sat` is removed, and masks are sought with `_mask` or no suffix and supported image extensions. A typical pair is `training/images/example_sat.jpg` and `training/masks/example_mask.png`. Review the prepared manifest rather than assuming unmatched files are included.

## Splits

The inherited discovery policy sorts the matched training pairs, shuffles them with `random.Random(seed)`, and reserves the first `max(1, round(N * val_fraction))` for validation. The final configuration uses seed 42 and a validation fraction of 0.1. The remaining pairs form training. Test pairs come from `eval/`; they are not the held-out training subset.

This describes the supplied code's local split policy. It does not establish that a particular directory matches an official dataset partition. Preserve any authoritative split lists and record how the local layout was produced. Changing file membership, ordering rules, or seed can change the partition.

```bash
oriweaveformer-prepare-data --config configs/experiments/oriweaveformer_massachusetts.yaml --data-root /path/to/massachusetts --output data/massachusetts/splits.json
```

Review the generated manifest and pass `--split-manifest data/massachusetts/splits.json` to later stages. Keep source/version metadata and checksums alongside it.

## Input processing

The loader decodes images as RGB and converts masks to binary road labels using values greater than 127. A multichannel mask is converted to grayscale first. Inputs are normalized with mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`.

Training uses random crops with a configured preference for crops containing roads, geometric transforms, and color/noise augmentation. The road-content preference makes a bounded number of attempts and is not a guarantee that every crop satisfies the requested road fraction. Images smaller than the training crop are padded; image padding reflects content, while mask padding is zero. Training validation uses center crops. Final evaluation uses native image resolution with 1024-pixel windows, stride 512, and D4 eight-way augmentation.

Keep raw imagery and masks unchanged. Optional local `raw/`, `interim/`, and `processed/` folders follow the dataset-stage convention; `--data-root` must point at the layout actually consumed by the loader. Dataset-specific result evidence belongs in `experiments/oriweaveformer_final/massachusetts/`.
