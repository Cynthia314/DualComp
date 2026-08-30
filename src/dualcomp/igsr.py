"""Clean-room reference IGSR semantics for a feature grid.

Paper-derived behavior from pages 8--10:
- Eq. 4 uses squared L2 difference from a caller-computed local average.
- Eq. 5 multiplies caller-normalized geometric and text scores using explicit beta.
- Eq. 6 greedily selects a caller-neighborhood candidate under strictly decreasing
  Chebyshev distance.

Reference API conventions, not paper settings:
- ascending numeric region labels define anchor traversal order;
- caller offset order resolves equal-scoring path candidates;
- complete paths are required, overlapping paths may repeat coordinates, and
  unused fixed-capacity output entries are zero padded with an explicit mask.

The paper describes a fully parallel implementation.  This transparent reference
uses Python control flow and does not claim that performance characteristic.
There are no learned parameters or hidden tunable constants.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as torch_functional


@dataclass(frozen=True)
class IGSRResult:
    """Fixed-capacity reference API output.

    Zero padding, the (-1, -1) coordinate sentinel, the validity mask, complete
    path retention, and possible repeated coordinates are interface conventions;
    pages 8--10 do not specify them.  Anchors follow ascending numeric values in
    the caller's region-label grid.
    """

    tokens: Tensor
    coordinates: Tensor
    valid_mask: Tensor
    sequence_lengths: Tensor
    anchors: Tensor
    geometric_saliency: Tensor
    structural_saliency: Tensor


def _require_floating_tensor(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values")


def _validate_beta(beta: Real) -> float:
    if isinstance(beta, bool) or not isinstance(beta, Real):
        raise TypeError("beta must be an explicit real scalar")
    beta_value = float(beta)
    if not math.isfinite(beta_value):
        raise ValueError("beta must be finite")
    return beta_value


def _validate_budget(budget: int) -> None:
    if isinstance(budget, bool) or not isinstance(budget, Integral):
        raise TypeError("budget must be an explicit integer")
    if budget <= 0:
        raise ValueError("budget must be positive")


def local_difference_saliency(
    features: Tensor,
    local_average: Tensor,
) -> Tensor:
    """Compute the squared L2 local difference supplied by Eq. 4.

    Both tensors must use [B, C, H, W] layout and have identical shape and
    device.  The caller computes the 3x3 local average because the paper
    does not specify boundary or padding behavior.
    """

    _require_floating_tensor(features, "features")
    _require_floating_tensor(local_average, "local_average")
    if features.ndim != 4:
        raise ValueError("features must have shape [B, C, H, W]")
    batch, channels, height, width = features.shape
    if batch == 0 or channels == 0 or height == 0 or width == 0:
        raise ValueError("features dimensions must all be non-zero")
    if local_average.shape != features.shape:
        raise ValueError("local_average must have the same shape as features")
    if local_average.device != features.device:
        raise ValueError("local_average and features must share a device")
    return (features - local_average).square().sum(dim=1)


def _validate_normalized_score_map(scores: Tensor, name: str) -> None:
    _require_floating_tensor(scores, name)
    if scores.ndim != 3:
        raise ValueError(f"{name} must have shape [B, H, W]")
    if scores.shape[0] == 0 or scores.shape[1] == 0 or scores.shape[2] == 0:
        raise ValueError(f"{name} dimensions must all be non-zero")


def structural_modulation(
    normalized_geometric_saliency: Tensor,
    normalized_text_relevance: Tensor,
    *,
    beta: Real,
) -> Tensor:
    """Apply Eq. 5 to caller-normalized geometric and text score maps."""

    _validate_normalized_score_map(
        normalized_geometric_saliency, "normalized_geometric_saliency"
    )
    _validate_normalized_score_map(
        normalized_text_relevance, "normalized_text_relevance"
    )
    if normalized_text_relevance.shape != normalized_geometric_saliency.shape:
        raise ValueError(
            "normalized_text_relevance must match "
            "normalized_geometric_saliency shape"
        )
    if normalized_text_relevance.device != normalized_geometric_saliency.device:
        raise ValueError("normalized score maps must share a device")
    beta_value = _validate_beta(beta)
    return normalized_geometric_saliency * (
        1.0 + beta_value * normalized_text_relevance
    )


def _ordered_region_labels(
    region_grid: Tensor,
    height: int,
    width: int,
) -> Tensor:
    """Validate labels and return their explicit ascending numeric order."""

    if not isinstance(region_grid, Tensor):
        raise TypeError("region_grid must be a torch.Tensor")
    if region_grid.ndim != 2 or tuple(region_grid.shape) != (height, width):
        raise ValueError(f"region_grid must have shape [{height}, {width}]")
    if (
        region_grid.dtype == torch.bool
        or region_grid.is_floating_point()
        or region_grid.is_complex()
    ):
        raise TypeError("region_grid must have an integer dtype")

    labels = torch.unique(region_grid, sorted=True)
    if labels.numel() == 0:
        raise ValueError("region_grid must contain at least one region")
    return labels


def select_region_anchors(
    geometric_saliency: Tensor,
    region_grid: Tensor,
) -> Tensor:
    """Select the unique saliency maximum in each caller-labeled region.

    Ascending numeric label order is a reference API convention that lets the
    caller explicitly determine anchor traversal.  Because the paper gives no
    maximum-tie rule, a tied regional maximum fails closed.
    """

    _require_floating_tensor(geometric_saliency, "geometric_saliency")
    if geometric_saliency.ndim != 3:
        raise ValueError("geometric_saliency must have shape [B, H, W]")
    batch, height, width = geometric_saliency.shape
    if batch == 0 or height == 0 or width == 0:
        raise ValueError("geometric_saliency dimensions must all be non-zero")

    ordered_labels = _ordered_region_labels(region_grid, height, width).to(
        device=geometric_saliency.device, dtype=torch.long
    )
    grid_labels = region_grid.to(
        device=geometric_saliency.device, dtype=torch.long
    ).flatten()
    flat_scores = geometric_saliency.flatten(start_dim=1)
    anchors = []
    for region_label in ordered_labels:
        in_region = grid_labels == region_label
        region_scores = flat_scores.masked_fill(
            ~in_region.unsqueeze(0), -torch.inf
        )
        maximum = region_scores.amax(dim=1, keepdim=True)
        is_maximum = in_region.unsqueeze(0) & (flat_scores == maximum)
        maximum_count = is_maximum.sum(dim=1)
        if bool((maximum_count != 1).any().item()):
            raise ValueError(
                "each region must have a unique geometric-saliency maximum; "
                "the paper does not specify anchor tie-breaking"
            )
        flat_index = is_maximum.to(dtype=torch.long).argmax(dim=1)
        row = torch.div(flat_index, width, rounding_mode="floor")
        column = flat_index.remainder(width)
        anchors.append(torch.stack((row, column), dim=1))
    return torch.stack(anchors, dim=1)


def _coordinate_tensor(
    coordinate: Sequence[int] | Tensor,
    *,
    name: str,
    device: torch.device,
    height: int,
    width: int,
) -> Tensor:
    raw = torch.as_tensor(coordinate, device=device)
    if raw.numel() != 2:
        raise ValueError(f"{name} must contain exactly two coordinates")
    raw = raw.reshape(2)
    if raw.is_floating_point():
        if not bool(torch.equal(raw, raw.round())):
            raise ValueError(f"{name} coordinates must be integers")
    point = raw.to(dtype=torch.long)
    row, column = int(point[0].item()), int(point[1].item())
    if not (0 <= row < height and 0 <= column < width):
        raise ValueError(f"{name} coordinate is outside the score grid")
    return point


def _validated_neighbor_offsets(
    neighbor_offsets: Tensor,
    *,
    device: torch.device,
) -> Tensor:
    if not isinstance(neighbor_offsets, Tensor):
        raise TypeError("neighbor_offsets must be a torch.Tensor")
    if neighbor_offsets.ndim != 2 or neighbor_offsets.shape[1] != 2:
        raise ValueError("neighbor_offsets must have shape [N, 2]")
    if neighbor_offsets.shape[0] == 0:
        raise ValueError("neighbor_offsets must contain at least one offset")
    if (
        neighbor_offsets.dtype == torch.bool
        or neighbor_offsets.is_floating_point()
        or neighbor_offsets.is_complex()
    ):
        raise TypeError("neighbor_offsets must have an integer dtype")
    if bool((neighbor_offsets == 0).all(dim=1).any().item()):
        raise ValueError("neighbor_offsets must not contain the zero offset")
    if torch.unique(neighbor_offsets, dim=0).shape[0] != neighbor_offsets.shape[0]:
        raise ValueError("neighbor_offsets must be unique")
    return neighbor_offsets.to(device=device, dtype=torch.long)


def trace_greedy_path(
    structural_saliency: Tensor,
    start: Sequence[int] | Tensor,
    goal: Sequence[int] | Tensor,
    neighbor_offsets: Tensor,
) -> Tensor:
    """Trace a greedy caller-defined-neighborhood path toward goal.

    Candidate construction and strict Chebyshev progress implement Eq. 6.
    Preserving caller offset order for equal scores is an explicit reference API
    convention because the paper does not define a tie rule.
    """

    _require_floating_tensor(structural_saliency, "structural_saliency")
    if structural_saliency.ndim != 2:
        raise ValueError("structural_saliency must have shape [H, W]")
    height, width = structural_saliency.shape
    if height == 0 or width == 0:
        raise ValueError("structural_saliency dimensions must be non-zero")

    offsets = _validated_neighbor_offsets(
        neighbor_offsets, device=structural_saliency.device
    )
    current = _coordinate_tensor(
        start,
        name="start",
        device=structural_saliency.device,
        height=height,
        width=width,
    )
    target = _coordinate_tensor(
        goal,
        name="goal",
        device=structural_saliency.device,
        height=height,
        width=width,
    )

    path = [current]
    while not bool(torch.equal(current, target)):
        current_distance = (current - target).abs().amax()
        candidates = current.unsqueeze(0) + offsets
        in_bounds = (
            (candidates[:, 0] >= 0)
            & (candidates[:, 0] < height)
            & (candidates[:, 1] >= 0)
            & (candidates[:, 1] < width)
        )
        candidates = candidates[in_bounds]
        candidate_distance = (candidates - target).abs().amax(dim=1)
        candidates = candidates[candidate_distance < current_distance]
        if candidates.shape[0] == 0:
            raise RuntimeError(
                "no Chebyshev-distance-decreasing candidate exists for "
                "the supplied neighbor_offsets"
            )
        candidate_scores = structural_saliency[
            candidates[:, 0], candidates[:, 1]
        ]
        current = candidates[candidate_scores.argmax()]
        path.append(current)

    return torch.stack(path, dim=0)


def recover_structure(
    features: Tensor,
    local_average: Tensor,
    normalized_geometric_saliency: Tensor,
    normalized_text_relevance: Tensor,
    region_grid: Tensor,
    neighbor_offsets: Tensor,
    *,
    beta: Real,
    budget: int,
) -> IGSRResult:
    """Run paper-derived IGSR semantics through a fail-closed reference API.

    Complete paths, repeated coordinates for overlapping walks, and zero-padded
    fixed-capacity output are API conventions rather than paper settings.  This
    implementation uses Python loops and is not the paper's claimed fully
    parallel performance implementation.
    """

    _require_floating_tensor(features, "features")
    if features.ndim != 4:
        raise ValueError("features must have shape [B, C, H, W]")
    batch, channels, height, width = features.shape
    if batch == 0 or channels == 0 or height == 0 or width == 0:
        raise ValueError("features dimensions must all be non-zero")
    _validate_budget(budget)

    expected_score_shape = (batch, height, width)
    for score_name, score_map in (
        ("normalized_geometric_saliency", normalized_geometric_saliency),
        ("normalized_text_relevance", normalized_text_relevance),
    ):
        if not isinstance(score_map, Tensor):
            raise TypeError(f"{score_name} must be a torch.Tensor")
        if tuple(score_map.shape) != expected_score_shape:
            raise ValueError(
                f"{score_name} must have shape [B, H, W] matching features"
            )
        if score_map.device != features.device:
            raise ValueError(f"features and {score_name} must share a device")

    offsets = _validated_neighbor_offsets(
        neighbor_offsets, device=features.device
    )
    ordered_region_labels = _ordered_region_labels(region_grid, height, width)
    region_count = int(ordered_region_labels.numel())
    if region_count > budget:
        raise ValueError(
            f"budget ({budget}) is smaller than the anchor count ({region_count})"
        )

    geometric = local_difference_saliency(features, local_average)
    structural = structural_modulation(
        normalized_geometric_saliency,
        normalized_text_relevance,
        beta=beta,
    )
    anchors = select_region_anchors(geometric, region_grid)

    routes = []
    required_lengths = []
    for batch_index in range(batch):
        route = anchors[batch_index, 0].unsqueeze(0)
        for anchor_index in range(1, region_count):
            segment = trace_greedy_path(
                structural[batch_index],
                route[-1],
                anchors[batch_index, anchor_index],
                offsets,
            )
            route = torch.cat((route, segment[1:]), dim=0)
        routes.append(route)
        required_lengths.append(route.shape[0])

    maximum_required = max(required_lengths)
    if maximum_required > budget:
        raise ValueError(
            f"budget ({budget}) is insufficient for topology-complete routes; "
            f"at least {maximum_required} entries are required"
        )

    padded_coordinates = []
    padded_tokens = []
    for batch_index, route in enumerate(routes):
        length = route.shape[0]
        token_sequence = features[
            batch_index, :, route[:, 0], route[:, 1]
        ].transpose(0, 1)
        padded_coordinates.append(
            torch_functional.pad(
                route,
                (0, 0, 0, budget - length),
                value=-1,
            )
        )
        padded_tokens.append(
            torch_functional.pad(
                token_sequence,
                (0, 0, 0, budget - length),
                value=0.0,
            )
        )

    coordinates = torch.stack(padded_coordinates, dim=0)
    tokens = torch.stack(padded_tokens, dim=0)
    sequence_lengths = torch.tensor(
        required_lengths, dtype=torch.long, device=features.device
    )
    positions = torch.arange(budget, device=features.device).unsqueeze(0)
    valid_mask = positions < sequence_lengths.unsqueeze(1)

    return IGSRResult(
        tokens=tokens,
        coordinates=coordinates,
        valid_mask=valid_mask,
        sequence_lengths=sequence_lengths,
        anchors=anchors,
        geometric_saliency=geometric,
        structural_saliency=structural,
    )


class ParameterFreeIGSR(nn.Module):
    """Parameter-free reference wrapper; intentionally not performance-parallel."""

    def forward(
        self,
        features: Tensor,
        local_average: Tensor,
        normalized_geometric_saliency: Tensor,
        normalized_text_relevance: Tensor,
        region_grid: Tensor,
        neighbor_offsets: Tensor,
        *,
        beta: Real,
        budget: int,
    ) -> IGSRResult:
        return recover_structure(
            features,
            local_average,
            normalized_geometric_saliency,
            normalized_text_relevance,
            region_grid,
            neighbor_offsets,
            beta=beta,
            budget=budget,
        )


__all__ = [
    "IGSRResult",
    "ParameterFreeIGSR",
    "local_difference_saliency",
    "recover_structure",
    "select_region_anchors",
    "structural_modulation",
    "trace_greedy_path",
]
