"""Typed, loader-independent tensor contract for selector windows."""

from dataclasses import dataclass

import torch
from torch import Tensor


MASKED_TARGET = -100
NULL_TARGET = -1


@dataclass
class SelectorBatch:
    """A padded batch of real frames and candidates.

    Candidate slots are batch padding only.  ``candidate_frame_indices`` maps
    every real candidate to a frame slot; target candidate indices are local to
    each frame and follow candidate slot order.  Stable candidate IDs belong in
    loader metadata and are intentionally absent from this contract.
    """

    candidate_values: Tensor
    candidate_validity: Tensor
    candidate_frame_indices: Tensor
    candidate_mask: Tensor
    frame_values: Tensor
    frame_validity: Tensor
    frame_mask: Tensor
    relative_time_seconds: Tensor
    targets: Tensor

    def validate(self, *, candidate_feature_dim: int = 12, frame_feature_dim: int | None = None) -> "SelectorBatch":
        cv, fv = self.candidate_values, self.frame_values
        if cv.ndim != 3 or cv.shape[-1] != candidate_feature_dim:
            raise ValueError(f"candidate_values must have shape [batch, candidates, {candidate_feature_dim}]")
        if self.candidate_validity.shape != cv.shape or self.candidate_validity.dtype != torch.bool:
            raise ValueError("candidate_validity must be a boolean tensor matching candidate_values")
        batch_size, candidate_count, _ = cv.shape
        if self.candidate_frame_indices.shape != (batch_size, candidate_count):
            raise ValueError("candidate_frame_indices must have shape [batch, candidates]")
        if self.candidate_frame_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("candidate_frame_indices must be an integer tensor")
        if self.candidate_mask.shape != (batch_size, candidate_count) or self.candidate_mask.dtype != torch.bool:
            raise ValueError("candidate_mask must be boolean with shape [batch, candidates]")
        if fv.ndim != 3 or fv.shape[0] != batch_size:
            raise ValueError("frame_values must have shape [batch, frames, features]")
        frame_count, actual_frame_dim = fv.shape[1], fv.shape[2]
        if frame_feature_dim is not None and actual_frame_dim != frame_feature_dim:
            raise ValueError(f"frame_values feature dimension must be {frame_feature_dim}")
        if self.frame_validity.shape != fv.shape or self.frame_validity.dtype != torch.bool:
            raise ValueError("frame_validity must be a boolean tensor matching frame_values")
        expected_frames = (batch_size, frame_count)
        if self.frame_mask.shape != expected_frames or self.frame_mask.dtype != torch.bool:
            raise ValueError("frame_mask must be boolean with shape [batch, frames]")
        if self.relative_time_seconds.shape != expected_frames:
            raise ValueError("relative_time_seconds must have shape [batch, frames]")
        if self.targets.shape != expected_frames or self.targets.dtype not in (torch.int32, torch.int64):
            raise ValueError("targets must be an integer tensor with shape [batch, frames]")
        tensors = (
            self.candidate_validity, self.candidate_frame_indices, self.candidate_mask,
            self.frame_values, self.frame_validity, self.frame_mask,
            self.relative_time_seconds, self.targets,
        )
        if any(t.device != cv.device for t in tensors):
            raise ValueError("all SelectorBatch tensors must be on the same device")
        if not torch.isfinite(cv[self.candidate_mask]).all():
            raise ValueError("real candidate values must be finite")
        expanded_frame_mask = self.frame_mask.unsqueeze(-1).expand_as(fv)
        if not torch.isfinite(fv[expanded_frame_mask]).all():
            raise ValueError("real frame values must be finite")
        if not torch.isfinite(self.relative_time_seconds[self.frame_mask]).all():
            raise ValueError("real frame relative times must be finite")

        for batch_index in range(batch_size):
            real_frames = self.frame_mask[batch_index]
            if real_frames.any() and not torch.any(
                torch.isclose(
                    self.relative_time_seconds[batch_index, real_frames],
                    torch.zeros((), device=cv.device, dtype=self.relative_time_seconds.dtype),
                )
            ):
                raise ValueError("each nonempty window must include a frame centered at relative time zero")
            for candidate_slot in torch.nonzero(self.candidate_mask[batch_index], as_tuple=False).flatten().tolist():
                frame_index = int(self.candidate_frame_indices[batch_index, candidate_slot])
                if frame_index < 0 or frame_index >= frame_count or not bool(real_frames[frame_index]):
                    raise ValueError("a real candidate references padding or a nonexistent frame")
            if torch.any(self.targets[batch_index, ~real_frames] != MASKED_TARGET):
                raise ValueError("padding frames must use target -100")
            for frame_index in torch.nonzero(real_frames, as_tuple=False).flatten().tolist():
                target = int(self.targets[batch_index, frame_index])
                if target < MASKED_TARGET or target not in (MASKED_TARGET, NULL_TARGET) and target < 0:
                    raise ValueError("targets must be -100, -1, or a nonnegative frame-local index")
                local_count = int(torch.sum(
                    self.candidate_mask[batch_index]
                    & (self.candidate_frame_indices[batch_index] == frame_index)
                ))
                if target >= local_count:
                    raise ValueError(
                        f"target {target} references no candidate in batch {batch_index}, frame {frame_index}"
                    )
        return self

    def to(self, device: torch.device | str) -> "SelectorBatch":
        return SelectorBatch(**{name: value.to(device) for name, value in vars(self).items()})
