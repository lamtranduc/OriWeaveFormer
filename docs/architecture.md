# Final OriWeaveFormer architecture

For the exhaustive layer-by-layer specification, tensor shapes, initialization,
parameter counts, gradient paths, and the imported MiT-B3 stage structure, see
[`architecture-complete-vi.md`](architecture-complete-vi.md).

`src/oriweaveformer/model.py` contains the complete supplied OriWeaveFormer implementation. The release preserves its tensor operations, submodule names, sampling layout, and state-dict keys. The description below documents that implementation.

The model returns a dictionary containing full-resolution single-channel `logits`. Its encoder is constructed with `segmentation_models_pytorch.encoders.get_encoder`; the final recipe selects `mit_b3` and uses the final four encoder feature maps.

## Orientation field and aggregation

An orientation head reads the finest retained encoder feature map. Two convolution, batch-normalization, and GELU blocks precede a three-channel output. The first two channels are normalized into `p` and `q`, interpreted as a doubled-angle representation, while a sigmoid produces confidence. The normalization denominator is `sqrt(p² + q² + 1e-6)`.

Residual orientation-conditioned feature enhancement is applied at the fine, middle, and deepest levels. Their default spans are `(4, 3, 8)`, with two sample locations per span index. Spacing is 4.0 for the fine and middle levels and 6.0 for the deepest level. Each branch fuses sampled context with its input through pointwise convolutions and gates the residual with confidence. A learned sigmoid scale starts at 0.1.

The sampling grid uses `theta = 0.5 * atan2(q, p)`, coordinates from `linspace(-1, 1)`, and offsets proportional to `spacing * index / width` for both axes. `grid_sample` uses bilinear interpolation, zero padding, and `align_corners=True`. The inherited grid is reshaped from `[B, S, H, W, 2]` to `[B, S*H, W, 2]`, and the sampled tensor is then reshaped directly to `[B, S, C, H, W]` before averaging. This exact layout convention is preserved for checkpoint and implementation parity; it must not be silently replaced with another axis interpretation.

The grid is computed from detached `p` and `q`. Thus the loss does not differentiate through their sampling coordinates. The same orientation outputs reach the decoder gate without detachment, so `p` and `q` still receive segmentation-loss gradients through the decoder. Confidence also participates in differentiable feature gating. The orientation head is not supervised by a separate orientation target.

## Decoder and objective

Each encoder scale is projected to 256 channels with a pointwise convolution, GroupNorm, and GELU, then resized to the finest retained feature resolution. The standard fusion branch concatenates the projected features in reverse scale order. A second branch predicts four softmax scale weights using the concatenated features and the three-channel orientation field, then sums the weighted projected features. The two branches are fused with a spatial convolution, GroupNorm, GELU, and dropout before a one-channel classifier. Bilinear interpolation restores the input image size using `align_corners=False`.

The objective is binary cross entropy with logits plus soft Dice loss. Dice pools the current minibatch's pixels and uses an epsilon of `1e-7`. No orientation loss, clDice loss, centerline loss, or additional topology objective is present. Under ordinary distributed gradient averaging, the loss is computed from each process's local minibatch; increasing world size does not turn that local Dice computation into a single globally pooled Dice ratio.

The package boundary and runnable research workflow are new. The architecture is not redesigned or corrected in this release. A change to the grid layout, orientation gradient paths, or module names would need its own explicit scientific and checkpoint-compatibility assessment.
