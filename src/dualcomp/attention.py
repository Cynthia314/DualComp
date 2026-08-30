"""Clean-room implementation of DualComp Equation 2."""

import math
from collections.abc import Sequence
from numbers import Integral
from typing import Union

import torch
from torch import Tensor


IndexInput = Union[Sequence[int], Tensor]


def _validated_patch_indices(indices: IndexInput, token_count: int, device: torch.device) -> Tensor:
    if isinstance(indices, Tensor):
        if indices.ndim != 1:
            raise ValueError("patch_key_indices must be one-dimensional")
        if indices.dtype != torch.long:
            raise TypeError("tensor patch_key_indices must have dtype torch.long")
        if indices.device != device:
            raise ValueError("patch_key_indices and projected keys must share a device")
        result = indices
    elif isinstance(indices, Sequence) and not isinstance(indices, (str, bytes)):
        values = list(indices)
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in values):
            raise TypeError("patch_key_indices must contain only integers")
        result = torch.tensor(values, dtype=torch.long, device=device)
    else:
        raise TypeError("patch_key_indices must be a one-dimensional long tensor or integer sequence")

    if result.numel() == 0:
        raise ValueError("patch_key_indices must select at least one patch")
    if bool(((result < 0) | (result >= token_count)).any()):
        raise IndexError("patch_key_indices contains an out-of-range index")
    if torch.unique(result).numel() != result.numel():
        raise ValueError("patch_key_indices must not contain duplicates")
    return result


def cls_to_patch_attention(
    projected_query: Tensor,
    projected_key: Tensor,
    *,
    cls_query_index: int,
    patch_key_indices: IndexInput,
) -> Tensor:
    """Compute the CLS-to-patch weights from Equation 2.

    ``projected_query`` and ``projected_key`` must already be projected by the
    caller and have shape ``[..., token, head_dim]``.  The leading dimensions
    (which may include batch and attention-head dimensions) must match.  The
    function returns one softmax distribution per leading-dimension element,
    with shape ``[..., selected_patches]``.

    The paper does not specify host sequence layout or head reduction, so the
    CLS query index and patch-key indices are required and per-head weights are
    returned unchanged.  Attention values are not an input to Equation 2: the
    subscript ``v`` in the paper's ``K_v`` denotes visual keys.
    """

    if not isinstance(projected_query, Tensor) or not isinstance(projected_key, Tensor):
        raise TypeError("projected_query and projected_key must be torch tensors")
    if projected_query.ndim < 2 or projected_key.ndim < 2:
        raise ValueError("projected tensors must have at least [token, head_dim] dimensions")
    if projected_query.ndim != projected_key.ndim:
        raise ValueError("projected query and key tensors must have the same rank")
    if projected_query.shape[:-2] != projected_key.shape[:-2]:
        raise ValueError("projected query and key tensors must share leading dimensions")
    if projected_query.shape[-1] != projected_key.shape[-1]:
        raise ValueError("projected query and key tensors must share head_dim")
    if projected_query.shape[-1] <= 0:
        raise ValueError("head_dim must be positive")
    if not torch.is_floating_point(projected_query) or not torch.is_floating_point(projected_key):
        raise TypeError("projected query and key tensors must be floating point")
    if projected_query.device != projected_key.device:
        raise ValueError("projected query and key tensors must share a device")
    if projected_query.dtype != projected_key.dtype:
        raise ValueError("projected query and key tensors must share a dtype")
    if isinstance(cls_query_index, bool) or not isinstance(cls_query_index, Integral):
        raise TypeError("cls_query_index must be an integer")

    cls_index = int(cls_query_index)
    query_count = projected_query.shape[-2]
    if cls_index < 0 or cls_index >= query_count:
        raise IndexError("cls_query_index is out of range")

    patch_indices = _validated_patch_indices(
        patch_key_indices,
        projected_key.shape[-2],
        projected_key.device,
    )
    cls_query = projected_query.select(dim=-2, index=cls_index)
    patch_keys = projected_key.index_select(dim=-2, index=patch_indices)
    logits = torch.matmul(cls_query.unsqueeze(-2), patch_keys.transpose(-1, -2)).squeeze(-2)
    logits = logits / math.sqrt(projected_query.shape[-1])
    return torch.softmax(logits, dim=-1)
