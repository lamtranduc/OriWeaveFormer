import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_encoder


def _group_count(channels: int, maximum: int = 32) -> int:
    groups = min(maximum, channels)
    while channels % groups:
        groups -= 1
    return groups


class OrientHead(nn.Module):
    """Predict normalized cos(2theta), sin(2theta), and confidence."""

    def __init__(self, in_channels: int, hidden_channels: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels), nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels), nn.GELU(),
            nn.Conv2d(hidden_channels, 3, 1),
        )

    def forward(self, feature: torch.Tensor):
        p, q, confidence = self.net(feature).chunk(3, dim=1)
        norm = torch.sqrt(p.square() + q.square() + 1e-6)
        return p / norm, q / norm, torch.sigmoid(confidence)


class FieldConditionedWeaveAgg(nn.Module):
    """Direction-aligned multi-point feature sampling and fusion."""

    def __init__(self, channels: int, span: int, spacing: float = 4.0) -> None:
        super().__init__()
        self.span, self.spacing = span, spacing
        hidden = max(32, channels // 2)
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * channels, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, channels, 1))

    @staticmethod
    def _resize(p, q, confidence, size):
        if p.shape[-2:] != size:
            p = F.interpolate(p, size=size, mode="bilinear", align_corners=False)
            q = F.interpolate(q, size=size, mode="bilinear", align_corners=False)
            confidence = F.interpolate(
                confidence, size=size, mode="bilinear", align_corners=False)
        norm = torch.sqrt(p.square() + q.square() + 1e-6)
        return p / norm, q / norm, confidence

    def _grid(self, p, q, height, width):
        theta = 0.5 * torch.atan2(q[:, 0], p[:, 0])
        cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, height, device=p.device, dtype=p.dtype),
            torch.linspace(-1, 1, width, device=p.device, dtype=p.dtype),
            indexing="ij")
        xs, ys = xs[None], ys[None]
        grids = []
        for index in range(1, self.span + 1):
            offset = self.spacing * index / width
            dx, dy = offset * cos_theta, offset * sin_theta
            grids.extend((torch.stack((xs + dx, ys + dy), dim=-1),
                          torch.stack((xs - dx, ys - dy), dim=-1)))
        return torch.stack(grids, dim=1).contiguous()

    def forward(self, feature, p, q, confidence):
        batch, channels, height, width = feature.shape
        p, q, confidence = self._resize(p, q, confidence, (height, width))
        sample_count = 2 * self.span
        grid = self._grid(p.detach(), q.detach(), height, width)
        samples = F.grid_sample(
            feature, grid.reshape(batch, sample_count * height, width, 2),
            mode="bilinear", padding_mode="zeros", align_corners=True)
        context = samples.reshape(
            batch, sample_count, channels, height, width).mean(dim=1)
        return feature + self.fuse(torch.cat((feature, context), 1)) * confidence


class ResidualOCFE(nn.Module):
    def __init__(self, channels: int, span: int, spacing: float = 4.0,
                 initial_scale: float = 0.1) -> None:
        super().__init__()
        self.aggregation = FieldConditionedWeaveAgg(channels, span, spacing)
        initial_scale = min(max(initial_scale, 1e-4), 1 - 1e-4)
        self.scale_logit = nn.Parameter(torch.tensor(
            math.log(initial_scale / (1 - initial_scale))))

    def forward(self, feature, p, q, confidence):
        aggregated = self.aggregation(feature, p, q, confidence)
        return feature + torch.sigmoid(self.scale_logit) * (aggregated - feature)


class ProjectNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels), nn.GELU())


class OrientationConditionedSegFormerDecoder(nn.Module):
    def __init__(self, in_channels: tuple[int, int, int, int],
                 decoder_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            ProjectNormAct(channel, decoder_dim) for channel in in_channels)
        joined = 4 * decoder_dim
        self.scale_gate = nn.Sequential(
            nn.Conv2d(joined + 3, decoder_dim, 1, bias=False),
            nn.GroupNorm(_group_count(decoder_dim), decoder_dim), nn.GELU(),
            nn.Conv2d(decoder_dim, 4, 1))
        self.concat_fuse = nn.Sequential(
            nn.Conv2d(joined, decoder_dim, 1, bias=False),
            nn.GroupNorm(_group_count(decoder_dim), decoder_dim), nn.GELU())
        self.output_fuse = nn.Sequential(
            nn.Conv2d(2 * decoder_dim, decoder_dim, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(decoder_dim), decoder_dim), nn.GELU(),
            nn.Dropout2d(dropout))
        self.classifier = nn.Conv2d(decoder_dim, 1, 1)
        nn.init.normal_(self.scale_gate[-1].weight, 0.0, 1e-3)
        nn.init.zeros_(self.scale_gate[-1].bias)
        nn.init.normal_(self.classifier.weight, 0.0, 0.01)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, features, p, q, confidence):
        size = features[0].shape[-2:]
        projected = []
        for projection, feature in zip(self.projections, features):
            feature = projection(feature)
            if feature.shape[-2:] != size:
                feature = F.interpolate(
                    feature, size=size, mode="bilinear", align_corners=False)
            projected.append(feature)
        orientation = torch.cat((p, q, confidence), 1)
        if orientation.shape[-2:] != size:
            orientation = F.interpolate(
                orientation, size=size, mode="bilinear", align_corners=False)
        joined = torch.cat(projected[::-1], 1)
        weights = torch.softmax(
            self.scale_gate(torch.cat((joined, orientation), 1)), dim=1)
        adaptive = sum(feature * weights[:, i:i + 1]
                       for i, feature in enumerate(projected))
        standard = self.concat_fuse(joined)
        return self.classifier(
            self.output_fuse(torch.cat((standard, adaptive), 1)))


class OriWeaveFormer(nn.Module):
    def __init__(self, encoder="mit_b3", pretrained="imagenet",
                 spans=(4, 3, 8), decoder_dim=256, dropout=0.1,
                 ocfe_initial_scale=0.1) -> None:
        super().__init__()
        self.encoder = get_encoder(
            encoder, in_channels=3, depth=5, weights=pretrained)
        channels = tuple(self.encoder.out_channels[-4:])
        fine, mid, _, deep = channels
        self.orientation = OrientHead(fine)
        self.ocfe_fine = ResidualOCFE(
            fine, spans[0], initial_scale=ocfe_initial_scale)
        self.ocfe_mid = ResidualOCFE(
            mid, spans[1], initial_scale=ocfe_initial_scale)
        self.ocfe_deep = ResidualOCFE(
            deep, spans[2], spacing=6.0, initial_scale=ocfe_initial_scale)
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
