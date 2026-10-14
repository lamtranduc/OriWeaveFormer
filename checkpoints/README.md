# Final OriWeaveFormer checkpoints

| File | Dataset and training setting | Epochs | Weights | SHA-256 |
|---|---|---:|---|---|
| `massachusetts.pt` | Massachusetts (512×512 crops, effective batch size 8) | 250 | EMA | `17ec066b16226e11e12120eb30c076d74549dffe242807a1681fc6f54260f469` |
| `deepglobe.pt` | DeepGlobe (512×512 crops, effective batch size 8) | 150 | EMA | `d1398faa4b3a8124528f56b34a13d53fb3a30fc8b475d5f4fdcb1f40ea27a090` |

Both files contain a dictionary with `ema`, configuration, checkpoint epoch, and validation metrics, matching the training protocol described in Section 4.2 of the paper. They load strictly into `OriWeaveFormer`.

These files are larger than GitHub's 100 MB regular-object limit. `.gitattributes` routes `checkpoints/*.pt` through Git LFS. Install Git LFS and run `git lfs install` before committing; alternatively publish checkpoints as release assets or in a model repository and remove them from the source commit.
