# Final OriWeaveFormer: DeepGlobe

The retained result uses final EMA weights from OriWeaveFormer trained on 512×512 crops for 150 epochs with effective batch size 8, as described in Section 4.2 of the paper. At the validation-selected optimal threshold **0.4458**, the test pooled results achieve Precision **83.25%**, Recall **83.95%**, F1 **83.60%** (0.836014), and IoU **71.82%** (0.718234), matching Table 1 in the paper. The labelled dataset consists of 6,226 satellite images partitioned into 5,000 training, 600 validation, and 626 testing pairs.

`metrics.json` preserves all reported numerical evaluation values, including fixed-threshold 0.5 results. `source_training_config.json` records the selected checkpoint configuration. `provenance.json` records the complete OriWeaveFormer stage lineage, historical sources, SHA256 digests and evidence limits. The shared protocol is in `../summary/evaluation_protocol.json`.

These are historical results imported into the reorganized release. Training and evaluation have not been rerun for this release, and the record does not assert numerical equivalence of the reorganized code. Dataset files and checkpoints are not included in this experiment directory.
