# Final OriWeaveFormer results

This experiment contains only the final OriWeaveFormer records for the two datasets. Values below are percentages, rounded from the exact fractions in the dataset `metrics.json` files. Precision, recall, F1 and IoU pool all test pixels before calculation.

| Dataset and checkpoint | Validation / test images | Threshold | Precision | Recall | F1 | IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Massachusetts, 250 epochs (512×512 crops) | 60 / 178 | 0.4670 | 78.4327 | 77.5141 | 77.9707 | 63.8951 |
| DeepGlobe, 150 epochs (512×512 crops) | 600 / 626 | 0.4458 | 83.2547 | 83.9510 | 83.6014 | 71.8234 |

Both use native-resolution inference with 1024-pixel tiles, stride 512, and tile-local D4 8TTA. Each threshold maximizes pooled F1 on the validation split over 0.25–0.75 at 0.0001 intervals, and is frozen for testing. Fixed-threshold 0.5 results are also retained in each dataset record.

Massachusetts uses the repository's 778-image subset, split into 540 training, 60 validation and 178 test images. DeepGlobe uses a deterministic split of 6226 publicly labelled training pairs into 5000 training, 600 validation and 626 test images. The DeepGlobe result must be identified as a custom holdout, not an official benchmark test score.

Both Massachusetts (250 epochs) and DeepGlobe (150 epochs) models are trained on 512×512 crops with effective batch size 8, following the protocol described in Section 4.2 of the paper.

These results were preserved from historical evaluations, with source result and checkpoint SHA256 digests recorded in each `provenance.json`. The reorganized code has not been used to regenerate these metrics. Exact historical package versions, per-image outputs, raw confusion counts and split manifests were not captured by the original evaluation JSONs. See `evaluation_protocol.json` for the complete archived inference and metric definitions.
