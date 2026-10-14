# Final OriWeaveFormer: Massachusetts

The retained result uses EMA weights from epoch 240 of the 250-epoch, 512-crop OriWeaveFormer run. At the validation-selected threshold **0.467**, test pooled F1 is **0.779706868622754** and IoU is **0.6389505361612936**. The split is 540 train / 60 validation / 178 test images from the repository's Massachusetts subset.

`metrics.json` preserves all reported numerical evaluation values, including fixed-threshold 0.5 results. `source_training_config.json` matches the selected checkpoint configuration. `provenance.json` records historical source paths, SHA256 digests, checkpoint selection and evidence limits. The shared protocol is in `../summary/evaluation_protocol.json`.

These are historical results imported into the reorganized release. Training and evaluation have not been rerun for this release, and the record does not assert numerical equivalence of the reorganized code. Dataset files and checkpoints are not included in this experiment directory.
