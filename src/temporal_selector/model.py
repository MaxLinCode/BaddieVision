"""Non-causal packed temporal transformer and frame-local selector loss."""

from dataclasses import dataclass

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


@dataclass
class SelectorOutput:
    candidate_logits: Tensor
    null_logits: Tensor
    encoded: EncodedSelectorBatch


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

        sequence_lengths = []
        for b in range(batch_size):
            length = 0
            for f in torch.nonzero(batch.frame_mask[b], as_tuple=False).flatten().tolist():
                length += 1 + int(torch.sum(batch.candidate_mask[b] & (batch.candidate_frame_indices[b] == f)))
            sequence_lengths.append(length)
        max_tokens = max(sequence_lengths, default=0)
        if max_tokens == 0:
            raise ValueError("a selector batch must contain at least one real frame")
        packed = candidate_tokens.new_zeros((batch_size, max_tokens, self.config.token_size))
        packed_times = batch.relative_time_seconds.new_zeros((batch_size, max_tokens))
        padding_mask = torch.ones((batch_size, max_tokens), dtype=torch.bool, device=device)
        frame_map = torch.full((batch_size, frame_count), -1, dtype=torch.long, device=device)
        candidate_map = torch.full((batch_size, candidate_count), -1, dtype=torch.long, device=device)

        for b in range(batch_size):
            position = 0
            for f in torch.nonzero(batch.frame_mask[b], as_tuple=False).flatten().tolist():
                frame_map[b, f] = position
                packed[b, position] = frame_tokens[b, f]
                packed_times[b, position] = batch.relative_time_seconds[b, f]
                padding_mask[b, position] = False
                position += 1
                slots = torch.nonzero(
                    batch.candidate_mask[b] & (batch.candidate_frame_indices[b] == f), as_tuple=False
                ).flatten()
                for slot_tensor in slots:
                    slot = int(slot_tensor)
                    candidate_map[b, slot] = position
                    packed[b, position] = candidate_tokens[b, slot]
                    packed_times[b, position] = batch.relative_time_seconds[b, f]
                    padding_mask[b, position] = False
                    position += 1

        types = torch.full((batch_size, max_tokens), self.CANDIDATE_TYPE, dtype=torch.long, device=device)
        for b, f in torch.nonzero(frame_map >= 0, as_tuple=False).tolist():
            types[b, frame_map[b, f]] = self.FRAME_TYPE
        packed = (
            packed
            + self.token_type_embedding(types)
            + self.continuous_time_embedding(packed_times.unsqueeze(-1))
        )
        # No causal mask and no ordinal positional embedding: candidates in a
        # frame remain permutation-equivariant while both temporal sides attend.
        encoded = self.transformer(packed, src_key_padding_mask=padding_mask)
        return EncodedSelectorBatch(encoded, padding_mask, packed_times, frame_map, candidate_map)


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
        batch.validate(
            candidate_feature_dim=self.config.candidate_feature_dim,
            frame_feature_dim=self.config.frame_feature_dim,
        )
        output = output or self(batch)
        losses: list[Tensor] = []
        for b, f in torch.nonzero(batch.frame_mask & (batch.targets != MASKED_TARGET), as_tuple=False).tolist():
            slots = torch.nonzero(
                batch.candidate_mask[b] & (batch.candidate_frame_indices[b] == f), as_tuple=False
            ).flatten()
            logits = torch.cat((output.candidate_logits[b, slots], output.null_logits[b, f].view(1)))
            target = int(batch.targets[b, f])
            resolved_target = len(slots) if target == NULL_TARGET else target
            if resolved_target < 0 or resolved_target >= logits.numel():
                raise ValueError("target does not resolve to a candidate belonging to its frame")
            losses.append(F.cross_entropy(logits.view(1, -1), torch.tensor([resolved_target], device=logits.device)))
        if losses:
            return torch.stack(losses).mean()
        # Keep a differentiable scalar for completely unsupervised batches.
        finite_candidates = output.candidate_logits.masked_fill(~batch.candidate_mask, 0.0)
        finite_nulls = output.null_logits.masked_fill(~batch.frame_mask, 0.0)
        return (finite_candidates.sum() + finite_nulls.sum()) * 0.0
