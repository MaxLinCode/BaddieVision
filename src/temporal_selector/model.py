"""Non-causal packed temporal transformer and frame-local selector loss."""

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .batch import MASKED_TARGET, NULL_TARGET, SelectorBatch
from .config import SelectorConfig


@dataclass
class EncodedSelectorBatch:
    tokens: Tensor
    padding_mask: Tensor
    packed_relative_time_seconds: Tensor
    frame_token_indices: Tensor
    candidate_token_indices: Tensor
    candidate_local_indices: Tensor


@dataclass
class SelectorOutput:
    candidate_logits: Tensor
    null_logits: Tensor
    encoded: EncodedSelectorBatch
    inplay_logits: Tensor | None = None
    rally_start_logits: Tensor | None = None
    rally_end_logits: Tensor | None = None


@dataclass
class JointLosses:
    total: Tensor
    selection: Tensor
    inplay: Tensor
    selection_frames: int | None
    inplay_frames: int | None
    rally_start: Tensor | None = None
    rally_end: Tensor | None = None


class _InputEncoder(nn.Module):
    def __init__(self, input_size: int, token_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, token_size),
            nn.GELU(),
            nn.Linear(token_size, token_size),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.network(values)


class TemporalShuttleEncoder(nn.Module):
    """Encode frame/candidate tokens without causal or ordinal position masks."""

    FRAME_TYPE = 0
    CANDIDATE_TYPE = 1

    def __init__(self, config: SelectorConfig) -> None:
        super().__init__()
        self.config = config
        self.candidate_encoder = _InputEncoder(config.candidate_feature_dim * 2, config.token_size)
        self.frame_encoder = (
            _InputEncoder(config.frame_feature_dim * 2, config.token_size)
            if config.frame_feature_dim else None
        )
        self.base_frame_token = nn.Parameter(torch.empty(config.token_size))
        nn.init.normal_(self.base_frame_token, std=0.02)
        self.token_type_embedding = nn.Embedding(2, config.token_size)
        self.continuous_time_embedding = nn.Sequential(
            nn.Linear(1, config.token_size),
            nn.GELU(),
            nn.Linear(config.token_size, config.token_size),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.token_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.feed_forward_size,
            dropout=config.dropout,
            activation=config.activation,
            layer_norm_eps=config.layer_norm_eps,
            batch_first=True,
            norm_first=config.norm_first,
        )
        norm = nn.LayerNorm(config.token_size, eps=config.layer_norm_eps) if config.final_norm else None
        self.transformer = nn.TransformerEncoder(
            layer, config.num_layers, norm=norm, enable_nested_tensor=False
        )

    def forward(self, batch: SelectorBatch) -> EncodedSelectorBatch:
        if not batch.validated:
            batch.validate(
                candidate_feature_dim=self.config.candidate_feature_dim,
                frame_feature_dim=self.config.frame_feature_dim,
            )
        device = batch.candidate_values.device
        batch_size, candidate_count, _ = batch.candidate_values.shape
        frame_count = batch.frame_values.shape[1]
        candidate_inputs = torch.cat(
            (batch.candidate_values, batch.candidate_validity.to(batch.candidate_values.dtype)), dim=-1
        )
        candidate_tokens = self.candidate_encoder(candidate_inputs)
        if self.frame_encoder is None:
            frame_tokens = self.base_frame_token.view(1, 1, -1).expand(batch_size, frame_count, -1)
        else:
            frame_inputs = torch.cat(
                (batch.frame_values, batch.frame_validity.to(batch.frame_values.dtype)), dim=-1
            )
            frame_tokens = self.frame_encoder(frame_inputs)

        if frame_count == 0 or (not batch.validated and not bool(batch.frame_mask.any())):
            raise ValueError("a selector batch must contain at least one real frame")
        safe_candidate_frames = batch.candidate_frame_indices.clamp(0, frame_count - 1)
        if batch.packed_frame_indices is not None:
            frame_map = batch.packed_frame_indices
            candidate_map = batch.packed_candidate_indices
            candidate_local_indices = batch.candidate_local_indices
            max_tokens = int(batch.packed_token_count)
        else:
            # Directly constructed model inputs retain a tensor-only fallback.
            # Production collation precomputes these immutable structural maps
            # once on CPU and transfers them with the batch.
            candidate_counts = torch.zeros(
                (batch_size, frame_count), dtype=torch.long, device=device
            )
            candidate_counts.scatter_add_(
                1, safe_candidate_frames, batch.candidate_mask.to(torch.long)
            )
            group_sizes = (candidate_counts + 1) * batch.frame_mask.to(torch.long)
            frame_positions = group_sizes.cumsum(dim=1) - group_sizes
            frame_map = frame_positions.masked_fill(~batch.frame_mask, -1)
            candidate_to_frame = batch.candidate_mask.unsqueeze(-1) & (
                safe_candidate_frames.unsqueeze(-1)
                == torch.arange(frame_count, device=device).view(1, 1, -1)
            )
            candidate_ranks = (
                candidate_to_frame.to(torch.long)
                .cumsum(dim=1)
                .gather(2, safe_candidate_frames.unsqueeze(-1))
                .squeeze(-1)
                - 1
            )
            candidate_map = (
                frame_positions.gather(1, safe_candidate_frames) + 1 + candidate_ranks
            ).masked_fill(~batch.candidate_mask, -1)
            candidate_local_indices = candidate_ranks.masked_fill(
                ~batch.candidate_mask, -1
            )
            max_tokens = frame_count + candidate_count
        packed = candidate_tokens.new_zeros((batch_size, max_tokens, self.config.token_size))
        packed_times = batch.relative_time_seconds.new_zeros((batch_size, max_tokens))
        padding_mask = torch.ones((batch_size, max_tokens), dtype=torch.bool, device=device)

        frame_batch, frame_slots = torch.nonzero(batch.frame_mask, as_tuple=True)
        frame_token_positions = frame_map[frame_batch, frame_slots]
        packed[frame_batch, frame_token_positions] = frame_tokens[frame_batch, frame_slots]
        packed_times[frame_batch, frame_token_positions] = batch.relative_time_seconds[
            frame_batch, frame_slots
        ]
        padding_mask[frame_batch, frame_token_positions] = False

        candidate_batch, candidate_slots = torch.nonzero(
            batch.candidate_mask, as_tuple=True
        )
        candidate_token_positions = candidate_map[candidate_batch, candidate_slots]
        packed[candidate_batch, candidate_token_positions] = candidate_tokens[
            candidate_batch, candidate_slots
        ]
        candidate_frame_slots = safe_candidate_frames[
            candidate_batch, candidate_slots
        ]
        packed_times[candidate_batch, candidate_token_positions] = (
            batch.relative_time_seconds[candidate_batch, candidate_frame_slots]
        )
        padding_mask[candidate_batch, candidate_token_positions] = False

        types = torch.full((batch_size, max_tokens), self.CANDIDATE_TYPE, dtype=torch.long, device=device)
        types[frame_batch, frame_token_positions] = self.FRAME_TYPE
        packed = (
            packed
            + self.token_type_embedding(types)
            + self.continuous_time_embedding(packed_times.unsqueeze(-1))
        )
        # No causal mask and no ordinal positional embedding: candidates in a
        # frame remain permutation-equivariant while both temporal sides attend.
        encoded = self.transformer(packed, src_key_padding_mask=padding_mask)
        return EncodedSelectorBatch(
            encoded,
            padding_mask,
            packed_times,
            frame_map,
            candidate_map,
            candidate_local_indices,
        )


class CandidateSelectionHead(nn.Module):
    def __init__(self, token_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(token_size, 1)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.projection(tokens).squeeze(-1)


class NullSelectionHead(nn.Module):
    def __init__(self, token_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(token_size, 1)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.projection(tokens).squeeze(-1)


class InPlayHead(nn.Module):
    def __init__(self, token_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(token_size, 1)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.projection(tokens).squeeze(-1)


class IsolatedRallyEncoder(nn.Module):
    """Temporal rally encoder that never consumes shared selector tokens."""

    def __init__(self, config: SelectorConfig, *, presence_only: bool) -> None:
        super().__init__()
        self.config = config
        self.presence_only = presence_only
        self.frame_encoder = (
            _InputEncoder(config.frame_feature_dim * 2, config.token_size)
            if config.frame_feature_dim
            else None
        )
        self.base_frame_token = nn.Parameter(torch.empty(config.token_size))
        nn.init.normal_(self.base_frame_token, std=0.02)
        if presence_only:
            self.present_evidence_token = nn.Parameter(torch.empty(config.token_size))
            nn.init.normal_(self.present_evidence_token, std=0.02)
            self.candidate_encoder = None
        else:
            self.candidate_encoder = _InputEncoder(
                config.candidate_feature_dim * 2, config.token_size
            )
        self.null_evidence_token = nn.Parameter(torch.empty(config.token_size))
        nn.init.normal_(self.null_evidence_token, std=0.02)
        self.fusion = nn.Linear(config.token_size * 2, config.token_size)
        self.continuous_time_embedding = nn.Sequential(
            nn.Linear(1, config.token_size),
            nn.GELU(),
            nn.Linear(config.token_size, config.token_size),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.token_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.feed_forward_size,
            dropout=config.dropout,
            activation=config.activation,
            layer_norm_eps=config.layer_norm_eps,
            batch_first=True,
            norm_first=config.norm_first,
        )
        norm = (
            nn.LayerNorm(config.token_size, eps=config.layer_norm_eps)
            if config.final_norm
            else None
        )
        self.transformer = nn.TransformerEncoder(
            layer, config.num_layers, norm=norm, enable_nested_tensor=False
        )

    def forward(
        self, batch: SelectorBatch, selected_candidate_slots: Tensor | None
    ) -> Tensor:
        batch_size, frame_count = batch.frame_mask.shape
        if self.frame_encoder is None:
            frames = self.base_frame_token.view(1, 1, -1).expand(
                batch_size, frame_count, -1
            )
        else:
            frame_inputs = torch.cat(
                (batch.frame_values, batch.frame_validity.to(batch.frame_values.dtype)),
                dim=-1,
            )
            frames = self.frame_encoder(frame_inputs)

        if self.presence_only:
            present = torch.zeros_like(batch.frame_mask)
            candidate_batch, candidate_slots = torch.nonzero(
                batch.candidate_mask, as_tuple=True
            )
            if candidate_slots.numel():
                present[candidate_batch, batch.candidate_frame_indices[
                    candidate_batch, candidate_slots
                ]] = True
            evidence = torch.where(
                present.unsqueeze(-1),
                self.present_evidence_token.view(1, 1, -1),
                self.null_evidence_token.view(1, 1, -1),
            )
        else:
            if selected_candidate_slots is None:
                raise ValueError("isolated content routing requires selected candidate slots")
            selected = selected_candidate_slots >= 0
            safe_slots = selected_candidate_slots.clamp_min(0)
            raw = batch.candidate_values.gather(
                1, safe_slots.unsqueeze(-1).expand(-1, -1, batch.candidate_values.shape[-1])
            )
            validity = batch.candidate_validity.gather(
                1, safe_slots.unsqueeze(-1).expand(-1, -1, batch.candidate_validity.shape[-1])
            )
            candidate_evidence = self.candidate_encoder(
                torch.cat((raw, validity.to(raw.dtype)), dim=-1)
            )
            evidence = torch.where(
                selected.unsqueeze(-1),
                candidate_evidence,
                self.null_evidence_token.view(1, 1, -1),
            )
        fused = frames + self.fusion(torch.cat((frames, evidence), dim=-1))
        fused = fused + self.continuous_time_embedding(
            batch.relative_time_seconds.unsqueeze(-1)
        )
        return self.transformer(fused, src_key_padding_mask=~batch.frame_mask)


class TemporalShuttleSelector(nn.Module):
    """Selector wrapper keeping reusable encoding separate from both heads."""

    def __init__(self, config: SelectorConfig | None = None) -> None:
        super().__init__()
        self.config = config or SelectorConfig()
        self.encoder = TemporalShuttleEncoder(self.config)
        self.selection_head = CandidateSelectionHead(self.config.token_size)
        self.null_head = NullSelectionHead(self.config.token_size)

    @staticmethod
    def _gather(tokens: Tensor, indices: Tensor) -> Tensor:
        safe = indices.clamp_min(0)
        gathered = tokens.gather(1, safe.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
        return gathered

    def forward(self, batch: SelectorBatch) -> SelectorOutput:
        encoded = self.encoder(batch)
        candidate_tokens = self._gather(encoded.tokens, encoded.candidate_token_indices)
        frame_tokens = self._gather(encoded.tokens, encoded.frame_token_indices)
        candidate_logits = self.selection_head(candidate_tokens)
        null_logits = self.null_head(frame_tokens)
        candidate_logits = candidate_logits.masked_fill(encoded.candidate_token_indices < 0, float("-inf"))
        null_logits = null_logits.masked_fill(encoded.frame_token_indices < 0, float("-inf"))
        return SelectorOutput(candidate_logits, null_logits, encoded)

    def loss(self, batch: SelectorBatch, output: SelectorOutput | None = None) -> Tensor:
        """Average independent candidate-plus-null cross entropy over supervised frames."""
        # ``forward`` performs full boundary validation. Avoid repeating its
        # data-dependent checks when the caller has already computed output;
        # on accelerators those checks synchronize device tensors with Python.
        output = self(batch) if output is None else output
        supervised = batch.frame_mask & (batch.targets != MASKED_TARGET)
        batch_size, frame_count = batch.frame_mask.shape
        candidate_count = batch.candidate_mask.shape[1]
        if bool(supervised.any()):
            # Candidate targets are frame-local. Build one padded class row per
            # frame and reserve the final class for NULL_TARGET, then evaluate
            # only sparse supervised rows in one cross-entropy call.
            frame_logits = output.candidate_logits.new_full(
                (batch_size, frame_count, candidate_count + 1), float("-inf")
            )
            candidate_batch, candidate_slots = torch.nonzero(
                batch.candidate_mask, as_tuple=True
            )
            candidate_frames = batch.candidate_frame_indices[
                candidate_batch, candidate_slots
            ]
            local_slots = output.encoded.candidate_local_indices[
                candidate_batch, candidate_slots
            ]
            frame_logits[candidate_batch, candidate_frames, local_slots] = (
                output.candidate_logits[candidate_batch, candidate_slots]
            )
            frame_logits[:, :, candidate_count] = output.null_logits
            resolved_targets = batch.targets.masked_fill(
                batch.targets == NULL_TARGET, candidate_count
            )
            return F.cross_entropy(
                frame_logits[supervised], resolved_targets[supervised]
            )
        finite_candidates = output.candidate_logits.masked_fill(
            ~batch.candidate_mask, 0.0
        )
        finite_nulls = output.null_logits.masked_fill(~batch.frame_mask, 0.0)
        return (finite_candidates.sum() + finite_nulls.sum()) * 0.0

    def joint_losses(
        self,
        batch: SelectorBatch,
        output: SelectorOutput | None = None,
        *,
        inplay_pos_weight: Tensor | None = None,
        selection_weight: float = 1.0,
        inplay_weight: float = 1.0,
        boundary_weight: float = 1.0,
        boundary_window_seconds: float = 1.0,
        boundary_start_pos_weight: Tensor | None = None,
        boundary_end_pos_weight: Tensor | None = None,
        boundary_aux_weight: float = 0.25,
        return_counts: bool = True,
    ) -> JointLosses:
        """Return independently normalized selection and binary rally losses."""
        if batch.inplay_targets is None:
            raise ValueError("joint training requires inplay_targets")
        output = output or self(batch)
        if output.inplay_logits is None:
            raise ValueError("joint model output is missing inplay_logits")
        if boundary_weight < 1.0 or boundary_window_seconds <= 0:
            raise ValueError("boundary weight must be >= 1 and its window must be positive")
        selection = TemporalShuttleSelector.loss(self, batch, output)
        supervised = batch.frame_mask & (batch.inplay_targets != MASKED_TARGET)
        inplay_losses = F.binary_cross_entropy_with_logits(
            output.inplay_logits,
            batch.inplay_targets.masked_fill(~supervised, 0).to(
                output.inplay_logits.dtype
            ),
            pos_weight=inplay_pos_weight,
            reduction="none",
        )
        frame_weights = torch.ones_like(inplay_losses)
        if boundary_weight > 1.0:
            for batch_index in range(batch.inplay_targets.shape[0]):
                transitions = torch.nonzero(
                    supervised[batch_index, :-1]
                    & supervised[batch_index, 1:]
                    & (
                        batch.inplay_targets[batch_index, :-1]
                        != batch.inplay_targets[batch_index, 1:]
                    ),
                    as_tuple=False,
                ).flatten()
                for transition in transitions.tolist():
                    boundary_time = (
                        batch.relative_time_seconds[batch_index, transition]
                        + batch.relative_time_seconds[batch_index, transition + 1]
                    ) / 2
                    distance = torch.abs(
                        batch.relative_time_seconds[batch_index] - boundary_time
                    )
                    taper = torch.clamp(
                        1.0 - distance / float(boundary_window_seconds), min=0.0
                    )
                    frame_weights[batch_index] = torch.maximum(
                        frame_weights[batch_index],
                        1.0 + (float(boundary_weight) - 1.0) * taper,
                    )
        # Padded transformer logits are allowed to be non-finite. Mask with
        # ``where`` before reduction: multiplying NaN by a zero weight would
        # still contaminate the scalar loss on MPS and IEEE-754 devices.
        effective_weights = torch.where(supervised, frame_weights, 0.0)
        weighted_losses = torch.where(
            supervised, inplay_losses * frame_weights, 0.0
        )
        inplay = weighted_losses.sum() / effective_weights.sum().clamp_min(1)
        selection_supervised = batch.frame_mask & (batch.targets != MASKED_TARGET)
        selection_frames = int(selection_supervised.sum()) if return_counts else None
        inplay_frames = int(supervised.sum()) if return_counts else None
        total = selection * float(selection_weight) + inplay * float(inplay_weight)
        start_loss = end_loss = None
        if output.rally_start_logits is not None or output.rally_end_logits is not None:
            if (
                output.rally_start_logits is None
                or output.rally_end_logits is None
                or batch.rally_start_targets is None
                or batch.rally_end_targets is None
            ):
                raise ValueError("boundary heads require paired logits and targets")

            def boundary_loss(logits: Tensor, targets: Tensor, pos_weight: Tensor | None) -> Tensor:
                mask = batch.frame_mask & (targets != MASKED_TARGET)
                values = F.binary_cross_entropy_with_logits(
                    logits,
                    targets.masked_fill(~mask, 0).to(logits.dtype),
                    pos_weight=pos_weight,
                    reduction="none",
                )
                return torch.where(mask, values, 0.0).sum() / mask.sum().clamp_min(1)

            start_loss = boundary_loss(
                output.rally_start_logits, batch.rally_start_targets, boundary_start_pos_weight
            )
            end_loss = boundary_loss(
                output.rally_end_logits, batch.rally_end_targets, boundary_end_pos_weight
            )
            total = total + float(boundary_aux_weight) * (start_loss + end_loss) / 2
        return JointLosses(
            total, selection, inplay, selection_frames, inplay_frames, start_loss, end_loss
        )


class JointRallyShuttleModel(TemporalShuttleSelector):
    """Joint binary rally-state and conditional shuttle-selection model."""

    CONDITIONING_MODES = frozenset(
        {
            "none",
            "soft_detached",
            "soft_joint",
            "presence_only",
            "hard_shared",
            "hard_isolated",
            "presence_isolated",
        }
    )

    def __init__(
        self,
        config: SelectorConfig | None = None,
        *,
        boundary_heads: bool = False,
        conditioning_mode: str = "none",
        force_candidate_routing: bool = False,
    ) -> None:
        super().__init__(config)
        if conditioning_mode not in self.CONDITIONING_MODES:
            choices = ", ".join(sorted(self.CONDITIONING_MODES))
            raise ValueError(
                f"unknown conditioning mode {conditioning_mode!r}; expected one of {choices}"
            )
        self.boundary_heads = bool(boundary_heads)
        self.conditioning_mode = conditioning_mode
        if force_candidate_routing and conditioning_mode != "hard_isolated":
            raise ValueError(
                "forced candidate routing is a diagnostic option for hard_isolated only"
            )
        self.force_candidate_routing = bool(force_candidate_routing)
        self.inplay_head = InPlayHead(self.config.token_size)
        shared_conditioning = self.conditioning_mode in {
            "soft_detached", "soft_joint", "presence_only", "hard_shared"
        }
        if shared_conditioning:
            self.null_evidence_token = nn.Parameter(
                torch.empty(self.config.token_size)
            )
            nn.init.normal_(self.null_evidence_token, std=0.02)
            if self.conditioning_mode == "presence_only":
                self.present_evidence_token = nn.Parameter(
                    torch.empty(self.config.token_size)
                )
                nn.init.normal_(self.present_evidence_token, std=0.02)
            self.conditioning_fusion = nn.Linear(
                self.config.token_size * 2, self.config.token_size
            )
        if self.conditioning_mode in {"hard_isolated", "presence_isolated"}:
            self.isolated_rally_encoder = IsolatedRallyEncoder(
                self.config,
                presence_only=self.conditioning_mode == "presence_isolated",
            )
        if self.boundary_heads:
            self.rally_start_head = InPlayHead(self.config.token_size)
            self.rally_end_head = InPlayHead(self.config.token_size)

    @classmethod
    def from_checkpoint(
        cls, checkpoint: Mapping[str, Any]
    ) -> "JointRallyShuttleModel":
        """Reconstruct a joint model without silently dropping conditioning."""
        state = checkpoint["model_state_dict"]
        recorded_mode = checkpoint.get("conditioning_mode")
        has_conditioning_state = any(
            key.startswith(
                (
                    "conditioning_fusion.",
                    "null_evidence_token",
                    "present_evidence_token",
                    "isolated_rally_encoder.",
                )
            )
            for key in state
        )
        if recorded_mode is None and has_conditioning_state:
            raise ValueError(
                "conditioned checkpoint is missing required conditioning_mode metadata"
            )
        mode = "none" if recorded_mode is None else str(recorded_mode)
        model = cls(
            SelectorConfig(**checkpoint["selector_config"]),
            boundary_heads=bool(checkpoint.get("boundary_heads", False)),
            conditioning_mode=mode,
            force_candidate_routing=bool(
                checkpoint.get("force_candidate_routing", False)
            ),
        )
        model.load_state_dict(state)
        return model

    def frame_selection_probabilities(
        self, output: SelectorOutput
    ) -> tuple[Tensor, Tensor]:
        """Return original-slot candidate and frame-local null probabilities."""
        batch_size, candidate_count = output.candidate_logits.shape
        frame_count = output.null_logits.shape[1]
        frame_logits = output.candidate_logits.new_full(
            (batch_size, frame_count, candidate_count + 1), float("-inf")
        )
        valid_candidates = output.encoded.candidate_token_indices >= 0
        candidate_batch, candidate_slots = torch.nonzero(
            valid_candidates, as_tuple=True
        )
        candidate_frames = (
            output.encoded.candidate_token_indices.new_zeros(
                output.encoded.candidate_token_indices.shape
            )
        )
        if candidate_slots.numel():
            # Packed candidates immediately follow their frame token. Resolve
            # the frame through the immutable packed maps without relying on
            # candidate input ordering.
            packed_positions = output.encoded.candidate_token_indices[
                candidate_batch, candidate_slots
            ]
            frame_maps = output.encoded.frame_token_indices[candidate_batch]
            candidate_frames[candidate_batch, candidate_slots] = (
                (frame_maps >= 0)
                & (packed_positions.unsqueeze(-1) > frame_maps)
            ).sum(dim=-1) - 1
            local_slots = output.encoded.candidate_local_indices[
                candidate_batch, candidate_slots
            ]
            frame_logits[
                candidate_batch,
                candidate_frames[candidate_batch, candidate_slots],
                local_slots,
            ] = output.candidate_logits[candidate_batch, candidate_slots]
        valid_frames = output.encoded.frame_token_indices >= 0
        frame_logits[:, :, candidate_count] = torch.where(
            valid_frames, output.null_logits, torch.zeros_like(output.null_logits)
        )
        probabilities = torch.softmax(frame_logits, dim=-1)
        probabilities = torch.where(
            valid_frames.unsqueeze(-1), probabilities, torch.zeros_like(probabilities)
        )
        candidate_probabilities = output.candidate_logits.new_zeros(
            output.candidate_logits.shape
        )
        if candidate_slots.numel():
            candidate_probabilities[candidate_batch, candidate_slots] = probabilities[
                candidate_batch,
                candidate_frames[candidate_batch, candidate_slots],
                output.encoded.candidate_local_indices[
                    candidate_batch, candidate_slots
                ],
            ]
        return candidate_probabilities, probabilities[:, :, candidate_count]

    def _conditioned_frame_tokens(
        self, output: SelectorOutput, frame_tokens: Tensor
    ) -> Tensor:
        candidate_probabilities, null_probabilities = (
            self.frame_selection_probabilities(output)
        )
        if self.conditioning_mode != "soft_joint":
            candidate_probabilities = candidate_probabilities.detach()
            null_probabilities = null_probabilities.detach()
        candidate_tokens = self._gather(
            output.encoded.tokens, output.encoded.candidate_token_indices
        )
        batch_size, frame_count = output.null_logits.shape
        evidence = frame_tokens.new_zeros(
            (batch_size, frame_count, self.config.token_size)
        )
        valid_candidates = output.encoded.candidate_token_indices >= 0
        candidate_batch, candidate_slots = torch.nonzero(
            valid_candidates, as_tuple=True
        )
        if candidate_slots.numel():
            packed_positions = output.encoded.candidate_token_indices[
                candidate_batch, candidate_slots
            ]
            frame_maps = output.encoded.frame_token_indices[candidate_batch]
            candidate_frames = (
                (frame_maps >= 0)
                & (packed_positions.unsqueeze(-1) > frame_maps)
            ).sum(dim=-1) - 1
            if self.conditioning_mode == "presence_only":
                values = self.present_evidence_token.view(1, -1).expand(
                    candidate_slots.shape[0], -1
                )
            else:
                values = candidate_tokens[candidate_batch, candidate_slots]
            evidence.index_put_(
                (candidate_batch, candidate_frames),
                values
                * candidate_probabilities[
                    candidate_batch, candidate_slots
                ].unsqueeze(-1),
                accumulate=True,
            )
        evidence = evidence + (
            null_probabilities.unsqueeze(-1)
            * self.null_evidence_token.view(1, 1, -1)
        )
        return frame_tokens + self.conditioning_fusion(
            torch.cat((frame_tokens, evidence), dim=-1)
        )

    def _hard_selected_candidate_slots(self, output: SelectorOutput) -> Tensor:
        """Return detached original candidate slots, or -1 for a null route."""
        candidate_probabilities, null_probabilities = (
            self.frame_selection_probabilities(output)
        )
        candidate_probabilities = candidate_probabilities.detach()
        null_probabilities = null_probabilities.detach()
        batch_size, frame_count = output.null_logits.shape
        selected = torch.full(
            (batch_size, frame_count),
            -1,
            dtype=torch.long,
            device=output.null_logits.device,
        )
        valid = output.encoded.candidate_token_indices >= 0
        candidate_batch, candidate_slots = torch.nonzero(valid, as_tuple=True)
        candidate_frames = torch.full_like(
            output.encoded.candidate_token_indices, -1
        )
        if candidate_slots.numel():
            packed_positions = output.encoded.candidate_token_indices[
                candidate_batch, candidate_slots
            ]
            frame_maps = output.encoded.frame_token_indices[candidate_batch]
            candidate_frames[candidate_batch, candidate_slots] = (
                (frame_maps >= 0)
                & (packed_positions.unsqueeze(-1) > frame_maps)
            ).sum(dim=-1) - 1
        for batch_index in range(batch_size):
            for frame_index in range(frame_count):
                slots = candidate_slots[
                    (candidate_batch == batch_index)
                    & (candidate_frames[batch_index, candidate_slots] == frame_index)
                ]
                if slots.numel():
                    best = slots[candidate_probabilities[batch_index, slots].argmax()]
                    if (
                        candidate_probabilities[batch_index, best]
                        > null_probabilities[batch_index, frame_index]
                    ):
                        selected[batch_index, frame_index] = best
        return selected

    @staticmethod
    def _first_available_candidate_slots(batch: SelectorBatch) -> Tensor:
        """Oracle-diagnostic routing for windows containing at most one candidate/frame."""
        batch_size, frame_count = batch.frame_mask.shape
        selected = torch.full(
            (batch_size, frame_count),
            -1,
            dtype=torch.long,
            device=batch.candidate_values.device,
        )
        candidate_batch, candidate_slots = torch.nonzero(
            batch.candidate_mask, as_tuple=True
        )
        for batch_index, slot in zip(
            candidate_batch.tolist(), candidate_slots.tolist()
        ):
            frame = int(batch.candidate_frame_indices[batch_index, slot])
            if selected[batch_index, frame] >= 0:
                raise ValueError(
                    "forced candidate routing requires at most one candidate per frame"
                )
            selected[batch_index, frame] = slot
        return selected

    def _hard_shared_frame_tokens(
        self, output: SelectorOutput, frame_tokens: Tensor, selected: Tensor
    ) -> Tensor:
        safe_slots = selected.clamp_min(0)
        candidate_tokens = self._gather(
            output.encoded.tokens, output.encoded.candidate_token_indices
        )
        evidence = candidate_tokens.gather(
            1, safe_slots.unsqueeze(-1).expand(-1, -1, candidate_tokens.shape[-1])
        )
        evidence = torch.where(
            (selected >= 0).unsqueeze(-1),
            evidence,
            self.null_evidence_token.view(1, 1, -1),
        )
        return frame_tokens + self.conditioning_fusion(
            torch.cat((frame_tokens, evidence), dim=-1)
        )

    def forward(self, batch: SelectorBatch) -> SelectorOutput:
        output = super().forward(batch)
        frame_tokens = self._gather(
            output.encoded.tokens, output.encoded.frame_token_indices
        )
        if self.conditioning_mode in {"soft_detached", "soft_joint", "presence_only"}:
            frame_tokens = self._conditioned_frame_tokens(output, frame_tokens)
        elif self.conditioning_mode == "hard_shared":
            selected = self._hard_selected_candidate_slots(output)
            frame_tokens = self._hard_shared_frame_tokens(output, frame_tokens, selected)
        elif self.conditioning_mode in {"hard_isolated", "presence_isolated"}:
            selected = (
                (
                    self._first_available_candidate_slots(batch)
                    if self.force_candidate_routing
                    else self._hard_selected_candidate_slots(output)
                )
                if self.conditioning_mode == "hard_isolated"
                else None
            )
            frame_tokens = self.isolated_rally_encoder(batch, selected)
        inplay_logits = self.inplay_head(frame_tokens).masked_fill(
            output.encoded.frame_token_indices < 0, float("-inf")
        )
        rally_start_logits = rally_end_logits = None
        if self.boundary_heads:
            padding = output.encoded.frame_token_indices < 0
            rally_start_logits = self.rally_start_head(frame_tokens).masked_fill(
                padding, float("-inf")
            )
            rally_end_logits = self.rally_end_head(frame_tokens).masked_fill(
                padding, float("-inf")
            )
        return SelectorOutput(
            output.candidate_logits,
            output.null_logits,
            output.encoded,
            inplay_logits,
            rally_start_logits,
            rally_end_logits,
        )

    def loss(
        self,
        batch: SelectorBatch,
        output: SelectorOutput | None = None,
        *,
        inplay_pos_weight: Tensor | None = None,
        selection_weight: float = 1.0,
        inplay_weight: float = 1.0,
    ) -> Tensor:
        return self.joint_losses(
            batch,
            output,
            inplay_pos_weight=inplay_pos_weight,
            selection_weight=selection_weight,
            inplay_weight=inplay_weight,
        ).total
