"""Candidate-independent offline rally segmentation model and tensor contract."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .batch import MASKED_TARGET
from .rally_audio import RALLY_AUDIO_FEATURE_NAMES
from .rally_features import (
    RALLY_FEATURE_SCHEMA,
    RALLY_FEATURE_VERSION,
    RALLY_FEATURE_VIEWS,
)


RALLY_FEATURE_DIM = RALLY_FEATURE_VIEWS["full"].dimension
RALLY_CONTEXT_DIRECTION = "offline_centered"
RALLY_MODEL_TYPE = "RallySegmenter"
RALLY_BOUNDARY_SUPERVISION = ("triangle_bce", "gaussian_qfl", "end_gaussian_qfl")


@dataclass
class RallyBatch:
    """Padded frame-only input for rally segmentation.

    This contract deliberately has no shuttle-candidate fields. ``frame_mask``
    distinguishes real frames from batch padding while ``frame_validity``
    describes missing individual visual and audio measurements.
    """

    frame_values: Tensor
    frame_validity: Tensor
    relative_time_seconds: Tensor
    frame_mask: Tensor
    inplay_targets: Tensor
    rally_start_targets: Tensor | None = None
    rally_end_targets: Tensor | None = None
    validated: bool = False

    def validate(self, *, frame_feature_dim: int = RALLY_FEATURE_DIM) -> "RallyBatch":
        values = self.frame_values
        if values.ndim != 3 or values.shape[-1] != frame_feature_dim:
            raise ValueError(
                "frame_values must have shape "
                f"[batch, frames, {frame_feature_dim}]"
            )
        batch_size, frame_count, _ = values.shape
        expected_frames = (batch_size, frame_count)
        if (
            self.frame_validity.shape != values.shape
            or self.frame_validity.dtype != torch.bool
        ):
            raise ValueError(
                "frame_validity must be a boolean tensor matching frame_values"
            )
        if (
            self.frame_mask.shape != expected_frames
            or self.frame_mask.dtype != torch.bool
        ):
            raise ValueError(
                "frame_mask must be boolean with shape [batch, frames]"
            )
        if self.relative_time_seconds.shape != expected_frames:
            raise ValueError(
                "relative_time_seconds must have shape [batch, frames]"
            )
        if (
            self.inplay_targets.shape != expected_frames
            or self.inplay_targets.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(
                "inplay_targets must be an integer tensor with shape [batch, frames]"
            )
        allowed = (
            (self.inplay_targets == MASKED_TARGET)
            | (self.inplay_targets == 0)
            | (self.inplay_targets == 1)
        )
        if not bool(torch.all(allowed)):
            raise ValueError("inplay_targets must contain only -100, 0, or 1")

        boundaries = (self.rally_start_targets, self.rally_end_targets)
        if any(target is not None for target in boundaries):
            if any(target is None for target in boundaries):
                raise ValueError("rally start/end targets must be supplied together")
            for name, target in zip(("start", "end"), boundaries):
                assert target is not None
                if target.shape != expected_frames or not target.is_floating_point():
                    raise ValueError(
                        f"rally {name} targets must be floating point with "
                        "shape [batch, frames]"
                    )
                valid = (target == MASKED_TARGET) | (
                    (target >= 0.0) & (target <= 1.0)
                )
                if not bool(torch.all(valid)):
                    raise ValueError(
                        f"rally {name} targets must contain -100 or values in [0, 1]"
                    )

        tensors = (
            self.frame_validity,
            self.relative_time_seconds,
            self.frame_mask,
            self.inplay_targets,
            *(target for target in boundaries if target is not None),
        )
        if any(tensor.device != values.device for tensor in tensors):
            raise ValueError("all RallyBatch tensors must be on the same device")
        expanded_mask = self.frame_mask.unsqueeze(-1).expand_as(values)
        if not bool(torch.isfinite(values[expanded_mask]).all()):
            raise ValueError("real frame values must be finite")
        if not bool(
            torch.isfinite(self.relative_time_seconds[self.frame_mask]).all()
        ):
            raise ValueError("real frame relative times must be finite")

        for batch_index in range(batch_size):
            real = self.frame_mask[batch_index]
            if not bool(real.any()):
                raise ValueError("every rally window must contain a real frame")
            centered = torch.isclose(
                self.relative_time_seconds[batch_index, real],
                torch.zeros(
                    (),
                    device=values.device,
                    dtype=self.relative_time_seconds.dtype,
                ),
            )
            if not bool(centered.any()):
                raise ValueError(
                    "each rally window must include a frame at relative time zero"
                )
            if bool(
                torch.any(
                    self.inplay_targets[batch_index, ~real] != MASKED_TARGET
                )
            ):
                raise ValueError("padding frames must use InPlay target -100")
            for target in boundaries:
                if target is not None and bool(
                    torch.any(target[batch_index, ~real] != MASKED_TARGET)
                ):
                    raise ValueError(
                        "padding frames must use boundary target -100"
                    )

        self.validated = True
        return self

    def to(self, device: torch.device | str) -> "RallyBatch":
        return RallyBatch(
            **{
                name: value.to(device) if isinstance(value, Tensor) else value
                for name, value in vars(self).items()
            }
        )


@dataclass(frozen=True)
class RallySegmenterConfig:
    frame_feature_dim: int = RALLY_FEATURE_DIM
    token_size: int = 128
    num_layers: int = 4
    num_attention_heads: int = 4
    feed_forward_size: int = 256
    activation: str = "gelu"
    dropout: float = 0.1
    norm_first: bool = True
    final_norm: bool = True
    layer_norm_eps: float = 1e-5
    audio_attention: bool = False

    def __post_init__(self) -> None:
        if self.frame_feature_dim <= 0:
            raise ValueError("frame_feature_dim must be positive")
        for name in (
            "token_size",
            "num_layers",
            "num_attention_heads",
            "feed_forward_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.token_size % self.num_attention_heads:
            raise ValueError("token_size must be divisible by num_attention_heads")
        if self.activation not in {"relu", "gelu"}:
            raise ValueError("activation must be 'relu' or 'gelu'")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.layer_norm_eps <= 0:
            raise ValueError("layer_norm_eps must be positive")
        if self.audio_attention and self.frame_feature_dim not in {
            view.dimension for view in RALLY_FEATURE_VIEWS.values() if view.uses_audio
        }:
            raise ValueError("audio attention requires the full visual and audio features")


@dataclass
class RallySegmenterOutput:
    inplay_logits: Tensor
    rally_start_logits: Tensor
    rally_end_logits: Tensor
    encoded_frames: Tensor


@dataclass
class RallyLosses:
    total: Tensor
    inplay: Tensor
    rally_start: Tensor
    rally_end: Tensor
    inplay_frames: int | None = None
    boundary_frames: int | None = None


class _FrameProjection(nn.Module):
    def __init__(self, input_size: int, token_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, token_size),
            nn.GELU(),
            nn.Linear(token_size, token_size),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.network(values)


class _BinaryFrameHead(nn.Module):
    def __init__(self, token_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(token_size, 1)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.projection(tokens).squeeze(-1)


class RallySegmenter(nn.Module):
    """Centered non-causal temporal model that consumes only frame context."""

    def __init__(self, config: RallySegmenterConfig | None = None) -> None:
        super().__init__()
        self.config = config or RallySegmenterConfig()
        audio_dim = len(RALLY_AUDIO_FEATURE_NAMES) if self.config.audio_attention else 0
        self.frame_projection = _FrameProjection(
            (self.config.frame_feature_dim - audio_dim) * 2, self.config.token_size
        )
        self.audio_projection = (
            _FrameProjection(audio_dim * 2, self.config.token_size) if audio_dim else None
        )
        self.continuous_time_embedding = nn.Sequential(
            nn.Linear(1, self.config.token_size),
            nn.GELU(),
            nn.Linear(self.config.token_size, self.config.token_size),
        )
        self.residual_fusion = nn.Sequential(
            nn.Linear(self.config.token_size * 2, self.config.token_size),
            nn.GELU(),
            nn.Linear(self.config.token_size, self.config.token_size),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.config.token_size,
            nhead=self.config.num_attention_heads,
            dim_feedforward=self.config.feed_forward_size,
            dropout=self.config.dropout,
            activation=self.config.activation,
            layer_norm_eps=self.config.layer_norm_eps,
            batch_first=True,
            norm_first=self.config.norm_first,
        )
        norm = (
            nn.LayerNorm(
                self.config.token_size, eps=self.config.layer_norm_eps
            )
            if self.config.final_norm
            else None
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            self.config.num_layers,
            norm=norm,
            enable_nested_tensor=False,
        )
        self.inplay_head = _BinaryFrameHead(self.config.token_size)
        self.rally_start_head = _BinaryFrameHead(self.config.token_size)
        self.rally_end_head = _BinaryFrameHead(self.config.token_size)

    def forward(self, batch: RallyBatch) -> RallySegmenterOutput:
        if not batch.validated:
            batch.validate(frame_feature_dim=self.config.frame_feature_dim)
        visual_dim = self.config.frame_feature_dim
        if self.audio_projection is not None:
            visual_dim -= len(RALLY_AUDIO_FEATURE_NAMES)
        frame_inputs = torch.cat(
            (
                batch.frame_values[..., :visual_dim],
                batch.frame_validity[..., :visual_dim].to(batch.frame_values.dtype),
            ),
            dim=-1,
        )
        frames = self.frame_projection(frame_inputs)
        time = self.continuous_time_embedding(
            batch.relative_time_seconds.unsqueeze(-1)
        )
        frames = frames + self.residual_fusion(torch.cat((frames, time), dim=-1))
        padding = ~batch.frame_mask
        tokens, token_padding = frames, padding
        if self.audio_projection is not None:
            audio_validity = batch.frame_validity[..., visual_dim:] & batch.frame_mask.unsqueeze(-1)
            audio_values = batch.frame_values[..., visual_dim:].masked_fill(~audio_validity, 0)
            audio_inputs = torch.cat((audio_values, audio_validity.to(audio_values.dtype)), dim=-1)
            audio_tokens = self.audio_projection(audio_inputs) + time
            tokens = torch.cat((frames, audio_tokens), dim=1)
            token_padding = torch.cat((padding, ~audio_validity.any(dim=-1)), dim=1)
        encoded = self.transformer(tokens, src_key_padding_mask=token_padding)[:, :frames.shape[1]]
        return RallySegmenterOutput(
            self.inplay_head(encoded).masked_fill(padding, float("-inf")),
            self.rally_start_head(encoded).masked_fill(
                padding, float("-inf")
            ),
            self.rally_end_head(encoded).masked_fill(padding, float("-inf")),
            encoded,
        )

    @staticmethod
    def _masked_bce(
        logits: Tensor,
        targets: Tensor,
        mask: Tensor,
        *,
        pos_weight: Tensor | None,
        quality_focal: bool = False,
    ) -> Tensor:
        safe_logits = torch.where(mask, logits, torch.zeros_like(logits))
        safe_targets = targets.masked_fill(~mask, 0).to(logits.dtype)
        values = F.binary_cross_entropy_with_logits(
            safe_logits,
            safe_targets,
            pos_weight=None if quality_focal else pos_weight,
            reduction="none",
        )
        if quality_focal:
            # Quality Focal Loss with beta=2 for continuous targets.
            values = values * (safe_logits.sigmoid() - safe_targets).square()
        return torch.where(mask, values, 0.0).sum() / mask.sum().clamp_min(1)

    def losses(
        self,
        batch: RallyBatch,
        output: RallySegmenterOutput | None = None,
        *,
        inplay_pos_weight: Tensor | None = None,
        boundary_start_pos_weight: Tensor | None = None,
        boundary_end_pos_weight: Tensor | None = None,
        boundary_aux_weight: float = 0.25,
        temporal_smoothing_weight: float = 0.0,
        quality_focal_start: bool = False,
        quality_focal_end: bool = False,
        return_counts: bool = True,
    ) -> RallyLosses:
        if boundary_aux_weight < 0:
            raise ValueError("boundary auxiliary weight cannot be negative")
        if not math.isfinite(temporal_smoothing_weight) or temporal_smoothing_weight < 0:
            raise ValueError("temporal smoothing weight must be finite and nonnegative")
        output = output or self(batch)
        inplay_mask = batch.frame_mask & (
            batch.inplay_targets != MASKED_TARGET
        )
        inplay = self._masked_bce(
            output.inplay_logits,
            batch.inplay_targets,
            inplay_mask,
            pos_weight=inplay_pos_weight,
        )

        if (
            batch.rally_start_targets is None
            or batch.rally_end_targets is None
        ):
            zero = inplay * 0.0
            start = end = zero
            boundary_frames = 0 if return_counts else None
        else:
            start_mask = batch.frame_mask & (
                batch.rally_start_targets != MASKED_TARGET
            )
            end_mask = batch.frame_mask & (
                batch.rally_end_targets != MASKED_TARGET
            )
            start = self._masked_bce(
                output.rally_start_logits,
                batch.rally_start_targets,
                start_mask,
                pos_weight=boundary_start_pos_weight,
                quality_focal=quality_focal_start,
            )
            end = self._masked_bce(
                output.rally_end_logits,
                batch.rally_end_targets,
                end_mask,
                pos_weight=boundary_end_pos_weight,
                quality_focal=quality_focal_end,
            )
            boundary_frames = (
                int((start_mask | end_mask).sum()) if return_counts else None
            )
        total = inplay + float(boundary_aux_weight) * (start + end) / 2
        if temporal_smoothing_weight:
            # Each row is a contiguous source window. Unowned targets are masked.
            same_state = batch.inplay_targets[:, 1:] == batch.inplay_targets[:, :-1]
            pair_mask = inplay_mask[:, 1:] & inplay_mask[:, :-1] & same_state
            probabilities = output.inplay_logits.masked_fill(~inplay_mask, 0.0).sigmoid()
            differences = (probabilities[:, 1:] - probabilities[:, :-1]).abs()
            smoothing = differences.masked_fill(~pair_mask, 0.0).sum()
            total = total + temporal_smoothing_weight * smoothing / pair_mask.sum().clamp_min(1)
        return RallyLosses(
            total=total,
            inplay=inplay,
            rally_start=start,
            rally_end=end,
            inplay_frames=int(inplay_mask.sum()) if return_counts else None,
            boundary_frames=boundary_frames,
        )

    @classmethod
    def from_checkpoint(
        cls, checkpoint: Mapping[str, Any]
    ) -> "RallySegmenter":
        required = {
            "model_type",
            "rally_feature_schema",
            "rally_feature_version",
            "rally_feature_view",
            "rally_feature_names",
            "dataset_fingerprint",
            "annotation_fingerprint",
            "context_direction",
            "split_manifest",
            "training_recipe",
            "chosen_epoch",
            "decoder_configuration",
            "rally_segmenter_config",
            "model_state_dict",
        }
        missing = sorted(required - set(checkpoint))
        if missing:
            raise ValueError(
                f"rally checkpoint is missing metadata: {', '.join(missing)}"
            )
        if checkpoint["model_type"] != RALLY_MODEL_TYPE:
            raise ValueError("checkpoint is not a RallySegmenter")
        if (
            checkpoint["rally_feature_schema"] != RALLY_FEATURE_SCHEMA
            or int(checkpoint["rally_feature_version"]) != RALLY_FEATURE_VERSION
        ):
            raise ValueError("rally checkpoint feature schema is incompatible")
        feature_view = str(checkpoint["rally_feature_view"])
        feature_names = tuple(map(str, checkpoint["rally_feature_names"]))
        feature_dim = int(checkpoint["rally_segmenter_config"]["frame_feature_dim"])
        if feature_view == "custom":
            if len(feature_names) != feature_dim:
                raise ValueError("custom checkpoint feature names/dimension are incompatible")
        elif feature_view not in RALLY_FEATURE_VIEWS:
            raise ValueError("rally checkpoint feature view is incompatible")
        else:
            registered = RALLY_FEATURE_VIEWS[feature_view]
            if feature_names != registered.names or feature_dim != registered.dimension:
                raise ValueError("rally checkpoint feature names/dimension are incompatible")
        if checkpoint["context_direction"] != RALLY_CONTEXT_DIRECTION:
            raise ValueError("rally checkpoint is not offline centered")
        model = cls(
            RallySegmenterConfig(**checkpoint["rally_segmenter_config"])
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return model


@dataclass(frozen=True)
class RallyTrainingRecipe:
    seed: int = 1729
    epochs: int = 25
    batch_size: int = 16
    sampling: str = "boundary_balanced"
    boundary_aux_weight: float = 0.25
    optimizer: str = "AdamW"
    learning_rate: float = 3e-4
    gradient_clip_norm: float = 1.0
    device: str = "mps"
    temporal_smoothing_weight: float = 0.0
    boundary_supervision: str = "triangle_bce"

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not isinstance(self.seed, int)
            or isinstance(self.seed, bool)
            or self.seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer")
        if self.boundary_supervision not in RALLY_BOUNDARY_SUPERVISION:
            raise ValueError("unsupported rally boundary supervision")
        if self.optimizer != "AdamW":
            raise ValueError("supported rally optimizer is AdamW")
        if self.sampling not in ("boundary_balanced", "uniform"):
            raise ValueError("unsupported rally sampling mode")
        if self.device not in ("cpu", "mps", "cuda"):
            raise ValueError("unsupported rally device")
        for name in ("learning_rate", "gradient_clip_norm", "boundary_aux_weight"):
            value = getattr(self, name)
            if (
                not math.isfinite(value)
                or value < 0
                or (name != "boundary_aux_weight" and value == 0)
            ):
                raise ValueError(
                    f"{name} must be finite and {'nonnegative' if name == 'boundary_aux_weight' else 'positive'}"
                )

        if not math.isfinite(self.temporal_smoothing_weight) or self.temporal_smoothing_weight < 0:
            raise ValueError("temporal smoothing weight must be finite and nonnegative")

    @property
    def gaussian_start(self) -> bool:
        return self.boundary_supervision == "gaussian_qfl"

    @property
    def gaussian_end(self) -> bool:
        return self.boundary_supervision != "triangle_bce"

    def recovery_settings(self) -> dict[str, Any]:
        settings = asdict(self)
        # Older runs have no smoothing setting in their recovery files.
        if not self.temporal_smoothing_weight:
            settings.pop("temporal_smoothing_weight")
        if self.boundary_supervision == "triangle_bce":
            settings.pop("boundary_supervision")
        return settings

    @classmethod
    def benchmark(cls) -> "RallyTrainingRecipe":
        """The historical 25-epoch MPS benchmark, preserved without overrides."""
        return cls()

    @classmethod
    def from_preset(
        cls, preset: str = "default", **overrides: Any
    ) -> "RallyTrainingRecipe":
        if preset not in ("default", "benchmark"):
            raise ValueError(f"unknown rally training preset: {preset}")
        recipe = cls(**overrides)
        if preset == "benchmark" and recipe != cls.benchmark():
            raise ValueError(
                "benchmark preset requires the exact historical training recipe"
            )
        return recipe


def rally_checkpoint(
    model: RallySegmenter,
    *,
    dataset_fingerprint: str,
    annotation_fingerprint: str,
    split_manifest: Mapping[str, Any],
    training_recipe: RallyTrainingRecipe,
    chosen_epoch: int,
    decoder_configuration: Mapping[str, Any],
    feature_view: str | None = None,
    calibration_fingerprints: Mapping[str, str] | None = None,
    pose_training_metadata: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a self-describing checkpoint without altering legacy formats."""
    if chosen_epoch <= 0:
        raise ValueError("chosen epoch must be positive")
    if not dataset_fingerprint or not annotation_fingerprint:
        raise ValueError("checkpoint fingerprints cannot be empty")
    if feature_view is None:
        feature_view = next(
            (
                name
                for name, view in RALLY_FEATURE_VIEWS.items()
                if view.dimension == model.config.frame_feature_dim
            ),
            "custom",
        )
    if feature_view == "custom":
        feature_names = tuple(
            f"custom_feature_{index}"
            for index in range(model.config.frame_feature_dim)
        )
    else:
        if feature_view not in RALLY_FEATURE_VIEWS:
            raise ValueError(f"unknown rally feature view: {feature_view}")
        registered_view = RALLY_FEATURE_VIEWS[feature_view]
        if model.config.frame_feature_dim != registered_view.dimension:
            raise ValueError("model dimension does not match checkpoint feature view")
        feature_names = registered_view.names
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "model_type": RALLY_MODEL_TYPE,
        "rally_segmenter_config": asdict(model.config),
        "rally_feature_schema": RALLY_FEATURE_SCHEMA,
        "rally_feature_version": RALLY_FEATURE_VERSION,
        "rally_feature_view": feature_view,
        "rally_feature_names": list(feature_names),
        "dataset_fingerprint": dataset_fingerprint,
        "annotation_fingerprint": annotation_fingerprint,
        "context_direction": RALLY_CONTEXT_DIRECTION,
        "split_manifest": dict(split_manifest),
        "training_recipe": asdict(training_recipe),
        "chosen_epoch": int(chosen_epoch),
        "decoder_configuration": dict(decoder_configuration),
        "calibration_fingerprints": dict(calibration_fingerprints or {}),
    }
    if pose_training_metadata is not None:
        payload["pose_training_metadata"] = dict(pose_training_metadata)
    if extra:
        overlap = set(payload).intersection(extra)
        if overlap:
            raise ValueError(
                f"extra checkpoint metadata overwrites required keys: {sorted(overlap)}"
            )
        payload.update(extra)
    return payload
