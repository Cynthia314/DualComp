"""Single-sample paper budget equations with caller-owned discretization."""

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Callable, Union

import torch
from torch import Tensor


Scalar = Union[Real, Tensor]


@dataclass(frozen=True)
class ContinuousTokenBudget:
    """The continuous paper allocation for exactly one sample."""

    max_tokens: int
    rho: float
    duality: float
    keep: float
    semantic: float
    geometric: float


@dataclass(frozen=True)
class TokenBudget:
    """A caller-chosen discrete allocation for exactly one sample."""

    keep: int
    semantic: int
    geometric: int


BudgetPolicy = Callable[[ContinuousTokenBudget], TokenBudget]


def _finite_scalar(name: str, value: Scalar) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar or one-element tensor")
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} tensor must contain exactly one value")
        if not torch.is_floating_point(value):
            raise TypeError(f"{name} tensor must have a floating-point dtype")
        scalar = float(value.detach().item())
    elif isinstance(value, Real):
        scalar = float(value)
    else:
        raise TypeError(f"{name} must be a real scalar or one-element tensor")
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _positive_max_tokens(max_tokens: int) -> int:
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, Integral):
        raise TypeError("max_tokens must be an integer")
    maximum = int(max_tokens)
    if maximum <= 0:
        raise ValueError("max_tokens must be positive")
    return maximum


def continuous_token_budget(
    *,
    max_tokens: int,
    rho: Scalar,
    duality: Scalar,
    rho_min: Scalar,
) -> ContinuousTokenBudget:
    """Compute one sample's paper allocation without integer rounding.

    The returned values follow ``n_keep = max_tokens * rho``,
    ``n_sem = n_keep * (1 - duality)``, and
    ``n_geo = n_keep * duality``.  ``rho_min`` remains mandatory because its
    value is not resolved by the method text.  ``rho`` and ``duality`` must be
    scalars or one-element tensors.  Batched Router outputs must be resolved one
    sample at a time by caller-owned integration code.
    """

    maximum = _positive_max_tokens(max_tokens)
    retention = _finite_scalar("rho", rho)
    minimum_retention = _finite_scalar("rho_min", rho_min)
    lambda_value = _finite_scalar("duality", duality)

    if not 0.0 <= minimum_retention <= 1.0:
        raise ValueError("rho_min must lie in [0, 1]")
    if not minimum_retention <= retention <= 1.0:
        raise ValueError("rho must lie in [rho_min, 1]")
    if not 0.0 <= lambda_value <= 1.0:
        raise ValueError("duality must lie in [0, 1]")

    n_keep = maximum * retention
    n_semantic = n_keep * (1.0 - lambda_value)
    n_geometric = n_keep * lambda_value
    return ContinuousTokenBudget(
        max_tokens=maximum,
        rho=retention,
        duality=lambda_value,
        keep=n_keep,
        semantic=n_semantic,
        geometric=n_geometric,
    )


def _validated_discrete_budget(budget: Any, max_tokens: int) -> TokenBudget:
    if not isinstance(budget, TokenBudget):
        raise TypeError("a discrete budget must be returned as TokenBudget")

    values = {}
    for name in ("keep", "semantic", "geometric"):
        value = getattr(budget, name)
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"TokenBudget.{name} must be an integer")
        integer = int(value)
        if integer < 0:
            raise ValueError(f"TokenBudget.{name} must be non-negative")
        values[name] = integer

    if values["semantic"] + values["geometric"] != values["keep"]:
        raise ValueError("TokenBudget semantic + geometric must equal keep")
    if values["keep"] > max_tokens:
        raise ValueError("TokenBudget.keep must not exceed max_tokens")
    return TokenBudget(**values)


def allocate_token_budget(
    *,
    max_tokens: int,
    rho: Scalar,
    duality: Scalar,
    rho_min: Scalar,
    budget_policy: BudgetPolicy,
) -> TokenBudget:
    """Apply and validate a mandatory single-sample discretization policy.

    No floor, round, or remainder convention is built into this library because
    the paper specifies continuous budgets but no integer conversion rule.  The
    library also imposes no minimum allocation for either stream.
    """

    continuous = continuous_token_budget(
        max_tokens=max_tokens,
        rho=rho,
        duality=duality,
        rho_min=rho_min,
    )
    if not callable(budget_policy):
        raise TypeError("budget_policy must be callable")
    return _validated_discrete_budget(
        budget_policy(continuous),
        continuous.max_tokens,
    )
