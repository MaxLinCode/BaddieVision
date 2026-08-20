"""Configuration for the temporal shuttle selector."""

from dataclasses import dataclass
from typing import Literal


ContextMode = Literal["candidates_only", "players_court", "full_context"]


@dataclass(frozen=True)
class SelectorConfig:
    """Model capacity and provisional frame-view configuration.

    ``frame_feature_dim`` is deliberately supplied by the dataset adapter.  The
    selector does not assign meanings or fixed widths to milestone-3 context
    views.
    """

    context_mode: ContextMode = "candidates_only"
    candidate_feature_dim: int = 12
    frame_feature_dim: int = 0
    token_size: int = 128
    num_layers: int = 4
    num_attention_heads: int = 4
    feed_forward_size: int = 256
    activation: str = "gelu"
    dropout: float = 0.1
    norm_first: bool = True
    final_norm: bool = True
    layer_norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.context_mode not in {"candidates_only", "players_court", "full_context"}:
            raise ValueError(f"unsupported context_mode: {self.context_mode!r}")
        if self.candidate_feature_dim != 12:
            raise ValueError("the selector candidate contract has exactly 12 numeric features")
        for name in ("token_size", "num_layers", "num_attention_heads", "feed_forward_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.token_size % self.num_attention_heads:
            raise ValueError("token_size must be divisible by num_attention_heads")
        if self.frame_feature_dim < 0:
            raise ValueError("frame_feature_dim cannot be negative")
        if self.context_mode == "candidates_only" and self.frame_feature_dim != 0:
            raise ValueError("candidates_only requires frame_feature_dim=0")
        if self.context_mode != "candidates_only" and self.frame_feature_dim == 0:
            raise ValueError(f"{self.context_mode} requires configured frame features")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.activation not in {"relu", "gelu"}:
            raise ValueError("activation must be 'relu' or 'gelu'")
        if self.layer_norm_eps <= 0:
            raise ValueError("layer_norm_eps must be positive")
