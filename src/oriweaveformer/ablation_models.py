import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_encoder

from .model import (
    _group_count,
    OrientHead,
    FieldConditionedWeaveAgg,
    ResidualOCFE,
    ProjectNormAct,
    OrientationConditionedSegFormerDecoder,
    OriWeaveFormer,
)


class StandardSegFormerDecoder(nn.Module):
    """Standard SegFormer MLP Decoder without orientation or scale gating."""

    def __init__(self, in_channels: tuple[int, int, int, int],
                 decoder_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            ProjectNormAct(channel, decoder_dim) for channel in in_channels)
        joined = 4 * decoder_dim
        self.concat_fuse = nn.Sequential(
            nn.Conv2d(joined, decoder_dim, 1, bias=False),
            nn.GroupNorm(_group_count(decoder_dim), decoder_dim), nn.GELU())
        self.output_fuse = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(decoder_dim), decoder_dim), nn.GELU(),
            nn.Dropout2d(dropout))
        self.classifier = nn.Conv2d(decoder_dim, 1, 1)
        nn.init.normal_(self.classifier.weight, 0.0, 0.01)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, features):
        size = features[0].shape[-2:]
        projected = []
        for projection, feature in zip(self.projections, features):
            feature = projection(feature)
            if feature.shape[-2:] != size:
                feature = F.interpolate(
                    feature, size=size, mode="bilinear", align_corners=False)
            projected.append(feature)
        joined = torch.cat(projected[::-1], 1)
        standard = self.concat_fuse(joined)
        return self.classifier(self.output_fuse(standard))


class DirectWeaveAgg(nn.Module):
    """OCFE without learnable residual gating (unmodulated direct aggregation)."""

    def __init__(self, channels: int, span: int, spacing: float = 4.0) -> None:
        super().__init__()
        self.aggregation = FieldConditionedWeaveAgg(channels, span, spacing)

    def forward(self, feature, p, q, confidence):
        return self.aggregation(feature, p, q, confidence)


# ==============================================================================
# 3 Ablation Models (matching Table 3 in the paper) + Full OriWeaveFormer
# ==============================================================================

class BaselineModel(nn.Module):
    """Method 1 in Paper: Baseline MiT-B3 + Standard SegFormer MLP Decoder."""

    def __init__(self, encoder="mit_b3", pretrained="imagenet",
                 decoder_dim=256, dropout=0.1) -> None:
        super().__init__()
        self.encoder = get_encoder(
            encoder, in_channels=3, depth=5, weights=pretrained)
        channels = tuple(self.encoder.out_channels[-4:])
        self.decoder = StandardSegFormerDecoder(channels, decoder_dim, dropout)

    def forward(self, image):
        pyramid = list(self.encoder(image)[-4:])
        logits = self.decoder(pyramid)
        logits = F.interpolate(
            logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return {"logits": logits}


class Model_NoRG(nn.Module):
    """Method 2 in Paper: Baseline + OFH + OCFE + ASG (w/o Learnable Residual Gating).

    Employs direct unscaled bilinear aggregation along tangents without the scalar residual gate sigma(alpha).
    """

    def __init__(self, encoder="mit_b3", pretrained="imagenet",
                 spans=(4, 3, 8), decoder_dim=256, dropout=0.1) -> None:
        super().__init__()
        self.encoder = get_encoder(
            encoder, in_channels=3, depth=5, weights=pretrained)
        channels = tuple(self.encoder.out_channels[-4:])
        fine, mid, _, deep = channels
        self.orientation = OrientHead(fine)
        self.ocfe_fine = DirectWeaveAgg(fine, spans[0])
        self.ocfe_mid = DirectWeaveAgg(mid, spans[1])
        self.ocfe_deep = DirectWeaveAgg(deep, spans[2], spacing=6.0)
        self.decoder = OrientationConditionedSegFormerDecoder(
            channels, decoder_dim, dropout)

    def forward(self, image):
        pyramid = list(self.encoder(image)[-4:])
        p, q, confidence = self.orientation(pyramid[0])
        pyramid[0] = self.ocfe_fine(pyramid[0], p, q, confidence)
        pyramid[1] = self.ocfe_mid(pyramid[1], p, q, confidence)
        pyramid[3] = self.ocfe_deep(pyramid[3], p, q, confidence)
        logits = self.decoder(pyramid, p, q, confidence)
        logits = F.interpolate(
            logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return {"logits": logits}


class Model_NoOCFE(nn.Module):
    """Method 3 in Paper: Baseline + OFH + ASG (w/o Directional Feature Enhancement).

    Bypasses the Residual OCFE modules entirely while retaining OFH and the Adaptive Scale Gate decoder.
    """

    def __init__(self, encoder="mit_b3", pretrained="imagenet",
                 decoder_dim=256, dropout=0.1) -> None:
        super().__init__()
        self.encoder = get_encoder(
            encoder, in_channels=3, depth=5, weights=pretrained)
        channels = tuple(self.encoder.out_channels[-4:])
        self.orientation = OrientHead(channels[0])
        self.decoder = OrientationConditionedSegFormerDecoder(
            channels, decoder_dim, dropout)

    def forward(self, image):
        pyramid = list(self.encoder(image)[-4:])
        p, q, confidence = self.orientation(pyramid[0])
        # Direct bypass - no OCFE enhancement
        logits = self.decoder(pyramid, p, q, confidence)
        logits = F.interpolate(
            logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return {"logits": logits}


def create_ablation_model(name: str, pretrained="imagenet") -> nn.Module:
    """Instantiate one of the 3 ablation models or the full OriWeaveFormer model.

    Args:
        name: Name of model variant:
              - 'baseline' / 'method1': Method 1 (Baseline MiT-B3 + SegFormer MLP)
              - 'wo_rg' / 'method2': Method 2 (w/o Learnable Residual Gating)
              - 'wo_ocfe' / 'method3': Method 3 (w/o Directional Feature Enhancement)
              - 'full' / 'oriweaveformer': Full proposed OriWeaveFormer model
        pretrained: Backbone pretraining weights (e.g. 'imagenet')
    """
    name = name.lower().replace("-", "_").replace(" ", "_")
    if name in ("baseline", "method1", "method_1"):
        return BaselineModel(pretrained=pretrained)
    elif name in ("wo_rg", "method2", "method_2", "no_rg"):
        return Model_NoRG(pretrained=pretrained)
    elif name in ("wo_ocfe", "method3", "method_3", "no_ocfe"):
        return Model_NoOCFE(pretrained=pretrained)
    elif name in ("full", "oriweaveformer", "proposed"):
        return OriWeaveFormer(pretrained=pretrained)
    else:
        raise ValueError(
            f"Unknown ablation model name '{name}'. Supported variants: "
            "['baseline' (Method 1), 'wo_rg' (Method 2), 'wo_ocfe' (Method 3), 'full']"
        )
