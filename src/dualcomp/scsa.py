"""A clean-room implementation of the paper's semantic token aggregator.

The implementation has no learned weights.  It consumes patch tokens laid out on
a rectangular grid, builds cosine-similarity clusters over a caller-supplied
predecessor graph, ranks those clusters with cumulative CLS attention, and emits
one token per selected cluster.

The paper describes ``tau(lambda)`` and ``theta_size(lambda)`` but does not give
their mappings or constants.  Consequently, this module deliberately provides no
defaults for ``tau`` or ``size_threshold``.  A caller may pass a scalar, one value
per batch item, or a callable that returns either form.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Callable, TypeAlias

import torch
from torch import Tensor, nn


ControlReturn: TypeAlias = float | int | Tensor
Control: TypeAlias = ControlReturn | Callable[[Tensor, Tensor], ControlReturn]
PredecessorNeighborhoods: TypeAlias = Sequence[Sequence[int]]


@dataclass(frozen=True)
class SCSAResult:
    """Result of semantic aggregation.

    Every selected-output field has second dimension exactly ``budget``.  A call
    fails if any batch item forms fewer clusters than requested, so no synthetic
    padding tokens are introduced and ``valid_mask`` is all true on success.

    Attributes:
        compressed_tokens: Selected representations, shaped ``[B, budget, D]``.
        valid_mask: All-true mask shaped ``[B, budget]``; no padding is emitted.
        importance: Cumulative CLS attention for every selected cluster.
        cluster_sizes: Number of input patches in every selected cluster.
        peak_indices: Peak input index for small clusters and ``-1`` for large ones.
        selected_cluster_ids: Per-image compact cluster identifier for each output.
        large_cluster_mask: True when the size-aware rule used the averaging branch.
        token_cluster_ids: Compact cluster identifier of every input token, ``[B, N]``.
        cluster_count: Number of clusters formed in each image, ``[B]``.
    """

    compressed_tokens: Tensor
    valid_mask: Tensor
    importance: Tensor
    cluster_sizes: Tensor
    peak_indices: Tensor
    selected_cluster_ids: Tensor
    large_cluster_mask: Tensor
    token_cluster_ids: Tensor
    cluster_count: Tensor


def _validate_inputs(
    tokens: Tensor,
    cls_attention: Tensor,
    height: int,
    width: int,
    budget: int,
) -> None:
    if tokens.ndim != 3:
        raise ValueError(f"tokens must have shape [B, N, D], got {tuple(tokens.shape)}")
    if cls_attention.ndim != 2:
        raise ValueError(
            "cls_attention must have shape [B, N], "
            f"got {tuple(cls_attention.shape)}"
        )
    if tokens.shape[:2] != cls_attention.shape:
        raise ValueError(
            "tokens and cls_attention must agree on B and N, got "
            f"{tuple(tokens.shape[:2])} and {tuple(cls_attention.shape)}"
        )
    if not tokens.is_floating_point():
        raise TypeError("tokens must use a floating-point dtype")
    if not cls_attention.is_floating_point():
        raise TypeError("cls_attention must use a floating-point dtype")
    if tokens.device != cls_attention.device:
        raise ValueError("tokens and cls_attention must be on the same device")

    for name, value in (("height", height), ("width", width)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if height * width != tokens.shape[1]:
        raise ValueError(
            f"height * width must equal N ({tokens.shape[1]}), got {height} * {width}"
        )
    if isinstance(budget, bool) or not isinstance(budget, Integral) or budget < 0:
        raise ValueError("budget must be a non-negative integer")
    if budget > tokens.shape[1]:
        raise ValueError(f"budget must not exceed N ({tokens.shape[1]})")

    if not bool(torch.isfinite(tokens).all()):
        raise ValueError("tokens must contain only finite values")
    if not bool(torch.isfinite(cls_attention).all()):
        raise ValueError("cls_attention must contain only finite values")
    if bool((cls_attention < 0).any()):
        raise ValueError("cls_attention must contain non-negative attention weights")


def _normalize_for_cosine(
    tokens: Tensor,
    predecessor_neighborhoods: list[list[int]],
) -> Tensor:
    """Normalize tokens used by graph edges without defining zero-vector cosine."""

    cosine_dtype = (
        torch.float32
        if tokens.dtype in (torch.float16, torch.bfloat16)
        else tokens.dtype
    )
    working = tokens.to(dtype=cosine_dtype)
    norms = torch.linalg.vector_norm(working, ord=2, dim=-1, keepdim=True)
    participates = [False] * tokens.shape[1]
    for token_index, candidates in enumerate(predecessor_neighborhoods):
        if candidates:
            participates[token_index] = True
            for candidate in candidates:
                participates[candidate] = True
    participation_mask = torch.tensor(
        participates, device=tokens.device, dtype=torch.bool
    ).unsqueeze(0)
    zero_norm = norms.squeeze(-1) == 0
    if bool((zero_norm & participation_mask).any()):
        raise ValueError(
            "tokens must have nonzero norm because cosine similarity is undefined "
            "for zero vectors"
        )
    if not bool(torch.isfinite(norms).all()):
        raise ValueError("token norms must be finite for cosine similarity")
    safe_norms = torch.where(norms == 0, torch.ones_like(norms), norms)
    return working / safe_norms


def _validate_predecessor_neighborhoods(
    predecessor_neighborhoods: PredecessorNeighborhoods,
    token_count: int,
) -> list[list[int]]:
    """Validate the caller's directed predecessor graph without inferring edges."""

    if isinstance(predecessor_neighborhoods, (str, bytes)) or not isinstance(
        predecessor_neighborhoods, Sequence
    ):
        raise TypeError("predecessor_neighborhoods must be a sequence of sequences")
    if len(predecessor_neighborhoods) != token_count:
        raise ValueError(
            "predecessor_neighborhoods must have length N "
            f"({token_count}), got {len(predecessor_neighborhoods)}"
        )

    validated: list[list[int]] = []
    for token_index, candidates in enumerate(predecessor_neighborhoods):
        if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
            raise TypeError(
                "each predecessor neighborhood must be a sequence of integers; "
                f"entry {token_index} has type {type(candidates).__name__}"
            )
        seen: set[int] = set()
        current: list[int] = []
        for candidate in candidates:
            if isinstance(candidate, bool) or not isinstance(candidate, Integral):
                raise TypeError(
                    "predecessor candidates must be integers; "
                    f"entry {token_index} contains {candidate!r}"
                )
            predecessor = int(candidate)
            if predecessor < 0 or predecessor >= token_index:
                raise ValueError(
                    "every predecessor candidate must satisfy 0 <= j < i; "
                    f"entry {token_index} contains {predecessor}"
                )
            if predecessor in seen:
                raise ValueError(
                    "predecessor candidates must be unique within each entry; "
                    f"entry {token_index} repeats {predecessor}"
                )
            seen.add(predecessor)
            current.append(predecessor)
        validated.append(current)
    return validated


def _resolve_control(
    control: Control,
    *,
    name: str,
    tokens: Tensor,
    cls_attention: Tensor,
    non_negative: bool,
) -> Tensor:
    """Evaluate and expand a scalar or per-image external control."""

    raw = control(tokens, cls_attention) if callable(control) else control
    if isinstance(raw, bool) or not isinstance(raw, (Real, Tensor)):
        raise TypeError(
            f"{name} must be a real number, tensor, or callable returning either"
        )
    if isinstance(raw, Tensor):
        if raw.device != tokens.device:
            raise ValueError(f"{name} tensor must be on the same device as tokens")
        if raw.dtype == torch.bool or raw.is_complex():
            raise TypeError(f"{name} tensor must contain real numeric values")
    try:
        values = torch.as_tensor(raw, device=tokens.device, dtype=torch.float64)
    except (TypeError, ValueError, RuntimeError) as error:
        raise TypeError(
            f"{name} must be a number, tensor, or callable returning either"
        ) from error

    batch = tokens.shape[0]
    if values.ndim == 0:
        values = values.expand(batch)
    elif values.ndim != 1 or values.shape[0] != batch:
        raise ValueError(
            f"{name} must resolve to a scalar or shape [B] ({batch},), "
            f"got {tuple(values.shape)}"
        )
    if not bool(torch.isfinite(values).all()):
        raise ValueError(f"{name} must contain only finite values")
    if non_negative and bool((values < 0).any()):
        raise ValueError(f"{name} must contain non-negative values")
    return values


def _cluster_one_image(
    normalized_tokens: Tensor,
    predecessor_neighborhoods: list[list[int]],
    tau: Tensor,
) -> tuple[list[list[int]], list[int]]:
    """Build connected components by linking each patch to one earlier neighbor."""

    token_count = normalized_tokens.shape[0]
    roots: list[int] = [0] * token_count

    for index, candidates in enumerate(predecessor_neighborhoods):
        parent = index
        if candidates:
            candidate_tensor = torch.tensor(
                candidates, device=normalized_tokens.device, dtype=torch.long
            )
            similarities = (
                normalized_tokens[candidate_tensor] * normalized_tokens[index]
            ).sum(dim=-1)
            maximum = torch.max(similarities)
            if bool(maximum > tau):
                maximizing = torch.nonzero(similarities == maximum, as_tuple=False).flatten()
                if maximizing.numel() != 1:
                    raise ValueError(
                        "cosine argmax is tied across predecessor candidates; "
                        "the paper does not specify a tie rule "
                        f"(token={index})"
                    )
                parent = candidates[int(maximizing[0])]
        roots[index] = index if parent == index else roots[parent]

    root_to_cluster: dict[int, int] = {}
    token_cluster_ids: list[int] = []
    members: list[list[int]] = []
    for token_index, root in enumerate(roots):
        if root not in root_to_cluster:
            root_to_cluster[root] = len(members)
            members.append([])
        cluster_id = root_to_cluster[root]
        token_cluster_ids.append(cluster_id)
        members[cluster_id].append(token_index)
    return members, token_cluster_ids


def spatially_contiguous_semantic_aggregate(
    tokens: Tensor,
    cls_attention: Tensor,
    height: int,
    width: int,
    budget: int,
    tau: Control,
    size_threshold: Control,
    predecessor_neighborhoods: PredecessorNeighborhoods,
) -> SCSAResult:
    """Compress a batch of spatial patch tokens without learned parameters.

    Args:
        tokens: Patch features shaped ``[B, N, D]`` in row-major grid order.
        cls_attention: Non-negative CLS-to-patch attention weights shaped ``[B, N]``.
            These are treated as weights exactly as supplied; this function does not
            apply softmax or renormalize them globally.
        height: Patch-grid height. ``height * width`` must equal ``N``.
        width: Patch-grid width.
        budget: Exact number of clusters retained per image.  The call fails if
            any image forms fewer clusters; no padding or clipping is performed.
        tau: Required cosine threshold.  This may be a scalar, a tensor shaped
            ``[B]``, or ``callable(tokens, cls_attention)`` returning either.  A
            callable can capture an external router value such as lambda.  A link is
            created only for cosine similarity strictly greater than this value.
        size_threshold: Required small/large cluster boundary, with the same scalar,
            per-batch, or callable forms as ``tau``.  Clusters of size less than or
            equal to this value retain their peak-attention patch; larger clusters
            use an attention-weighted mean.
        predecessor_neighborhoods: Required length-``N`` sequence.  Entry ``i``
            supplies the only tokens that patch ``i`` may link to.  Every candidate
            must be a unique integer satisfying ``0 <= j < i``.  The paper leaves
            its neighborhood construction unspecified, so this function never
            derives edges from ``height`` or ``width``.

    Returns:
        An :class:`SCSAResult`.  Selected clusters are ordered by descending
        cumulative attention.  A tie that would affect selection or output order
        raises because the paper supplies no tie-breaking rule.

    Notes:
        Each patch can link only to an earlier patch supplied by the caller.
        Following those links produces components in that explicit graph.  A
        selected large cluster with zero total attention raises an error because the
        paper defines no fallback for its attention-weighted average.  Zero-norm
        tokens are rejected because their cosine similarity is undefined.
    """

    _validate_inputs(tokens, cls_attention, height, width, budget)
    tau_by_image = _resolve_control(
        tau,
        name="tau",
        tokens=tokens,
        cls_attention=cls_attention,
        non_negative=False,
    )
    size_by_image = _resolve_control(
        size_threshold,
        name="size_threshold",
        tokens=tokens,
        cls_attention=cls_attention,
        non_negative=True,
    )
    validated_neighborhoods = _validate_predecessor_neighborhoods(
        predecessor_neighborhoods, tokens.shape[1]
    )

    batch, token_count, feature_size = tokens.shape
    slots = int(budget)
    compressed = tokens.new_zeros((batch, slots, feature_size))
    valid_mask = torch.zeros((batch, slots), device=tokens.device, dtype=torch.bool)
    importance = cls_attention.new_zeros((batch, slots))
    cluster_sizes = torch.zeros((batch, slots), device=tokens.device, dtype=torch.long)
    peak_indices = torch.full(
        (batch, slots), -1, device=tokens.device, dtype=torch.long
    )
    selected_cluster_ids = torch.full(
        (batch, slots), -1, device=tokens.device, dtype=torch.long
    )
    large_cluster_mask = torch.zeros(
        (batch, slots), device=tokens.device, dtype=torch.bool
    )
    token_cluster_ids = torch.empty(
        (batch, token_count), device=tokens.device, dtype=torch.long
    )
    cluster_count = torch.zeros((batch,), device=tokens.device, dtype=torch.long)

    normalized = _normalize_for_cosine(tokens, validated_neighborhoods)
    for batch_index in range(batch):
        members, labels = _cluster_one_image(
            normalized[batch_index],
            validated_neighborhoods,
            tau_by_image[batch_index],
        )
        token_cluster_ids[batch_index] = torch.tensor(
            labels, device=tokens.device, dtype=torch.long
        )
        cluster_count[batch_index] = len(members)
        if len(members) < slots:
            raise ValueError(
                "budget exceeds the number of clusters formed for a batch item; "
                "the paper does not define padding "
                f"(batch={batch_index}, budget={slots}, clusters={len(members)})"
            )

        cluster_importance = torch.stack(
            [
                cls_attention[batch_index, member_indices].sum()
                for member_indices in members
            ]
        )
        full_ranking = torch.argsort(cluster_importance, descending=True)
        ranking = full_ranking[:slots]
        if slots:
            selected_scores = cluster_importance[ranking]
            selected_tie = selected_scores.numel() > 1 and bool(
                (selected_scores[1:] == selected_scores[:-1]).any()
            )
            boundary_tie = slots < len(members) and bool(
                selected_scores[-1] == cluster_importance[full_ranking[slots]]
            )
            if selected_tie or boundary_tie:
                raise ValueError(
                    "cluster-importance tie affects selection or output order; "
                    "the paper does not specify a tie rule "
                    f"(batch={batch_index})"
                )

        for output_index, cluster_tensor in enumerate(ranking):
            cluster_id = int(cluster_tensor)
            member_indices = members[cluster_id]
            index_tensor = torch.tensor(
                member_indices, device=tokens.device, dtype=torch.long
            )
            member_attention = cls_attention[batch_index, index_tensor]
            cluster_size = len(member_indices)
            is_large = cluster_size > float(size_by_image[batch_index])
            peak_index = -1

            if is_large:
                summary_dtype = torch.promote_types(tokens.dtype, cls_attention.dtype)
                if summary_dtype in (torch.float16, torch.bfloat16):
                    summary_dtype = torch.float32
                member_tokens = tokens[batch_index, index_tensor].to(
                    dtype=summary_dtype
                )
                weights = member_attention.to(dtype=summary_dtype)
                total_weight = weights.sum()
                if bool(total_weight > 0):
                    representation = (
                        member_tokens * weights.unsqueeze(-1)
                    ).sum(dim=0) / total_weight
                    representation = representation.to(dtype=tokens.dtype)
                else:
                    raise ValueError(
                        "selected large cluster has zero total CLS attention; "
                        "the paper does not specify a fallback representation "
                        f"(batch={batch_index}, cluster={cluster_id})"
                    )
            else:
                maximum_attention = torch.max(member_attention)
                maximizing = torch.nonzero(
                    member_attention == maximum_attention, as_tuple=False
                ).flatten()
                if maximizing.numel() != 1:
                    raise ValueError(
                        "small-cluster peak attention is tied; "
                        "the paper does not specify a tie rule "
                        f"(batch={batch_index}, cluster={cluster_id})"
                    )
                peak_index = member_indices[int(maximizing[0])]
                representation = tokens[batch_index, peak_index]

            compressed[batch_index, output_index] = representation
            valid_mask[batch_index, output_index] = True
            importance[batch_index, output_index] = cluster_importance[cluster_id]
            cluster_sizes[batch_index, output_index] = cluster_size
            peak_indices[batch_index, output_index] = peak_index
            selected_cluster_ids[batch_index, output_index] = cluster_id
            large_cluster_mask[batch_index, output_index] = is_large

    return SCSAResult(
        compressed_tokens=compressed,
        valid_mask=valid_mask,
        importance=importance,
        cluster_sizes=cluster_sizes,
        peak_indices=peak_indices,
        selected_cluster_ids=selected_cluster_ids,
        large_cluster_mask=large_cluster_mask,
        token_cluster_ids=token_cluster_ids,
        cluster_count=cluster_count,
    )


class SpatiallyContiguousSemanticAggregator(nn.Module):
    """Stateless ``nn.Module`` wrapper around the functional SCSA API."""

    def forward(
        self,
        tokens: Tensor,
        cls_attention: Tensor,
        height: int,
        width: int,
        budget: int,
        tau: Control,
        size_threshold: Control,
        predecessor_neighborhoods: PredecessorNeighborhoods,
    ) -> SCSAResult:
        return spatially_contiguous_semantic_aggregate(
            tokens=tokens,
            cls_attention=cls_attention,
            height=height,
            width=width,
            budget=budget,
            tau=tau,
            size_threshold=size_threshold,
            predecessor_neighborhoods=predecessor_neighborhoods,
        )


__all__ = [
    "Control",
    "PredecessorNeighborhoods",
    "SCSAResult",
    "SpatiallyContiguousSemanticAggregator",
    "spatially_contiguous_semantic_aggregate",
]
