# Evaluation protocol

Training validation and final evaluation serve different purposes. Training uses deterministic center crops and a fixed 0.5 threshold for economical checkpoint selection. Final evaluation uses the full native-resolution images. Their scores should retain distinct labels in research notes and tables.

## Final inference

The final protocol processes native images with 1024-by-1024 sliding windows and stride 512. Overlapping predictions are combined into a full-image probability map. Each window uses the eight transforms of the square's dihedral group: rotations and reflected rotations. Predictions are mapped back to the original orientation before averaging. Output probabilities align to the native image extent; this is not a resize of the full image to 1024 pixels.

The training crop size and evaluation window size are separate settings. As specified in Section 4.2 and Section 4.3 of the paper, models are trained on 512×512 crops (250 epochs for Massachusetts, 150 epochs for DeepGlobe) and evaluated on native-resolution imagery using 1024×1024 sliding windows (stride 512) with full D4 8-way test-time augmentation (8-TTA).

## Threshold and metrics

Threshold selection uses validation labels only. The chosen threshold is frozen before scoring the held-out test partition. Test labels must not guide the threshold, a checkpoint choice, or repeated selection among evaluation recipes.

For binary predictions, pool true positives (`TP`), false positives (`FP`), and false negatives (`FN`) over all pixels in the evaluated split. Report:

| Metric | Definition |
| --- | --- |
| Precision | `TP / (TP + FP)` |
| Recall | `TP / (TP + FN)` |
| F1 | `2 TP / (2 TP + FP + FN)` |
| IoU | `TP / (TP + FP + FN)` |

The implementation uses a small numerical epsilon for zero denominators. These are pooled pixel metrics, not an unweighted mean of per-image metrics. Masks are binary road labels under the loader's threshold convention. They are not centerline or connectivity scores.

## Evidence to retain

Keep the JSON output, command, resolved configuration, split manifest, checkpoint hash and epoch, selected weights, threshold selection rule, threshold, window size, stride, augmentation, precision mode, and execution environment. `--device`, `--amp`, and `--weights` allow evaluation execution and weight selection; use `oriweaveformer-evaluate --help` for available values.

Historical metric JSON files supplied with the release are source evidence, not outputs reproduced during packaging. Their stage and protocol metadata remain authoritative for those measurements. Inspect the dataset records under `experiments/oriweaveformer_final/` before comparing a historical row with a new run. The included exporter builds a table from stored records; it does not evaluate a checkpoint.

The record must distinguish a validation score used for threshold selection from a test score obtained after selection. An internal DeepGlobe partition is not an official hidden-label challenge evaluation. Massachusetts results require the actual supplied evaluation split and its provenance to be reported.
