# DeepGlobe Roads

This release consumes a locally acquired set of labeled DeepGlobe road image/mask pairs. No dataset payloads or redistribution rights are included. Record the authoritative acquisition source, version or snapshot, license, and preferred citation for the actual files before publication; the supplied OriWeaveFormer source establishes a filename and partition convention only.

## Expected layout

```text
deepglobe/
  train/
    example_sat.jpg
    example_mask.png
    ...
```

Point `--data-root` at `deepglobe/`, or at an unambiguous directory containing the intended labeled `train/` directory. The loader can also discover the pair files directly under the root. Each `*_sat.jpg` must have the matching `*_mask.png`. Review the generated manifest for unmatched or missing inputs.

## Internal split

The inherited policy sorts paired satellite filenames, shuffles the pairs with `random.Random(seed)`, and partitions by position:

| Split | Positions after shuffling |
| --- | --- |
| Train | `pairs[:5000]` |
| Validation | `pairs[5000:5600]` |
| Test | `pairs[5600:]` |

The final configuration uses seed 42. This is an internal partition of the available labeled image collection. It is not a submission to the official hidden-label challenge test set. The final test count depends on the files actually present, and a smaller or different collection can invalidate comparisons with the historical records.

```bash
oriweaveformer-prepare-data --config configs/experiments/oriweaveformer_deepglobe.yaml --data-root /path/to/deepglobe --output data/deepglobe/splits.json
```

Inspect the manifest and use `--split-manifest data/deepglobe/splits.json` for base training, 1024-crop fine-tuning, and evaluation. Keep the same membership across these stages. Record file checksums and acquisition details alongside the manifest; matching a count is insufficient to establish the same split.

## Processing and stages

Images are decoded as RGB. Masks are converted to grayscale if needed and binarized with `mask > 127`. Images are normalized using mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`.

Base training uses 512-pixel crops. The separate fine-tuning configuration uses 1024-pixel crops and requires an explicit base checkpoint. Training crops use a bounded road-content preference plus geometric and color/noise augmentation. Training validation uses deterministic center crops. Final evaluation operates at native image resolution with 1024-pixel sliding windows, stride 512, and D4 eight-way augmentation; threshold selection uses only validation data.

Keep raw imagery and labels unchanged. Optional `raw/`, `interim/`, and `processed/` folders follow the dataset-stage convention. Base and fine-tuning evidence both belong in `experiments/oriweaveformer_final/deepglobe/` under the single final OriWeaveFormer experiment family.
