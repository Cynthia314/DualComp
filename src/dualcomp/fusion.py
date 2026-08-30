"""Dual-stream fusion described in DualComp, Section 3.2.

This module deliberately contains no projection or normalization layer.  Inputs
are expected to already share a feature space, as required by the paper's
feature-level fusion rule.
"""

from numbers import Real
from typing import Union

import torch
from torch import Tensor


Scalar = Union[Real, Tensor]


def _duality_weight(duality: Scalar, tokens: Tensor) -> Union[float, Tensor]:
    """Validate ``duality`` and shape it for explicit batch-wise weighting."""

    if isinstance(duality, bool):
        raise TypeError("duality must be a real scalar or floating-point tensor")

    if isinstance(duality, Tensor):
        if not torch.is_floating_point(duality):
            raise TypeError("tensor duality must have a floating-point dtype")
        if duality.device != tokens.device:
            raise ValueError("tensor duality and token tensors must share a device")
        if duality.dtype != tokens.dtype:
            raise ValueError("tensor duality and token tensors must share a dtype")

        batch_shape = tokens.shape[:-2]
        if duality.ndim == 0:
            weight = duality
        elif tuple(duality.shape) == tuple(batch_shape) and batch_shape:
            weight = duality.reshape(*batch_shape, 1, 1)
        else:
            expected = "a scalar" if not batch_shape else f"a scalar or shape {tuple(batch_shape)}"
            raise ValueError(f"tensor duality must be {expected}")

        if not bool(torch.isfinite(weight).all()):
            raise ValueError("duality must be finite")
        if bool(((weight < 0) | (weight > 1)).any()):
            raise ValueError("duality must lie in [0, 1]")
        return weight

    if not isinstance(duality, Real):
        raise TypeError("duality must be a real scalar or floating-point tensor")

    weight = float(duality)
    if not torch.isfinite(torch.tensor(weight)):
        raise ValueError("duality must be finite")
    if not 0.0 <= weight <= 1.0:
        raise ValueError("duality must lie in [0, 1]")
    return weight


def lambda_weighted_concatenation(
    semantic_tokens: Tensor,
    geometric_tokens: Tensor,
    duality: Scalar,
) -> Tensor:
    """Apply the paper's exact weighted-concatenation fusion rule.

    The last two dimensions of each input are interpreted as
    ``[token, feature]``.  All preceding dimensions are batch dimensions.
    Fusion is therefore::

        cat(((1 - duality) * semantic_tokens,
             duality * geometric_tokens), dim=-2)

    A tensor ``duality`` may be scalar or have exactly the shared batch shape.
    Requiring an exact batch shape avoids an ambiguous implicit broadcast.
    """

    if not isinstance(semantic_tokens, Tensor) or not isinstance(geometric_tokens, Tensor):
        raise TypeError("semantic_tokens and geometric_tokens must be torch tensors")
    if semantic_tokens.ndim < 2 or geometric_tokens.ndim < 2:
        raise ValueError("token tensors must have at least [token, feature] dimensions")
    if not torch.is_floating_point(semantic_tokens) or not torch.is_floating_point(
        geometric_tokens
    ):
        raise TypeError("token tensors must have floating-point dtypes")
    if semantic_tokens.device != geometric_tokens.device:
        raise ValueError("semantic and geometric token tensors must share a device")
    if semantic_tokens.dtype != geometric_tokens.dtype:
        raise ValueError("semantic and geometric token tensors must share a dtype")
    if semantic_tokens.ndim != geometric_tokens.ndim:
        raise ValueError("semantic and geometric token tensors must have the same rank")
    if semantic_tokens.shape[:-2] != geometric_tokens.shape[:-2]:
        raise ValueError("semantic and geometric token tensors must share batch dimensions")
    if semantic_tokens.shape[-1] != geometric_tokens.shape[-1]:
        raise ValueError("semantic and geometric token tensors must share feature width")

    weight = _duality_weight(duality, semantic_tokens)
    return torch.cat(
        ((1.0 - weight) * semantic_tokens, weight * geometric_tokens),
        dim=-2,
    )
