"""Paper-bounded clean-room wiring for the DualComp duality-aware router.

The paper leaves the compact instruction encoder, shared MLP, and both Sigmoid
heads structurally unspecified.  This module therefore requires callers to
inject all four modules and only validates their interfaces and final ranges.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class DualityAwareRouter(nn.Module):
    """Validate and connect caller-provided paper-level router components.

    Args:
        instruction_encoder: Module called as
            ``instruction_encoder(text_embeddings, text_attention_mask)``. It
            must return ``[batch, instruction_dim]``.
        shared_mlp: Module that maps the instruction representation to
            ``[batch, shared_dim]``.
        lambda_head: Independent caller-provided Sigmoid head. It must return
            final lambda values shaped ``[batch]`` and already in ``[0, 1]``.
        rho_head: Independent caller-provided Sigmoid head. It must return final
            rho values shaped ``[batch]`` and already in ``[rho_min, 1]``.
        instruction_dim: Explicit expected instruction-representation width.
        shared_dim: Explicit expected shared-representation width.
        rho_min: Explicit inclusive lower bound used only to validate rho.

    The wrapper supplies no pooling, adapter, MLP, activation, head, output
    mapping, initialization, or architecture-size defaults.
    """

    def __init__(
        self,
        instruction_encoder: nn.Module,
        shared_mlp: nn.Module,
        lambda_head: nn.Module,
        rho_head: nn.Module,
        instruction_dim: int,
        shared_dim: int,
        rho_min: float,
    ) -> None:
        super().__init__()

        modules = {
            "instruction_encoder": instruction_encoder,
            "shared_mlp": shared_mlp,
            "lambda_head": lambda_head,
            "rho_head": rho_head,
        }
        for name, module in modules.items():
            if not isinstance(module, nn.Module):
                raise TypeError(f"{name} must be an nn.Module")
        if lambda_head is rho_head:
            raise ValueError("lambda_head and rho_head must be independent module objects")

        self.instruction_dim = self._positive_dimension(
            "instruction_dim", instruction_dim
        )
        self.shared_dim = self._positive_dimension("shared_dim", shared_dim)
        self._rho_min = self._validated_rho_min(rho_min)

        self.instruction_encoder = instruction_encoder
        self.shared_mlp = shared_mlp
        self.lambda_head = lambda_head
        self.rho_head = rho_head

    @staticmethod
    def _positive_dimension(name: str, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be a positive integer")
        value = int(value)
        if value <= 0:
            raise ValueError(f"{name} must be greater than zero")
        return value

    @staticmethod
    def _validated_rho_min(value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("rho_min must be a real number")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("rho_min must be finite")
        if not 0.0 <= value <= 1.0:
            raise ValueError("rho_min must lie in [0, 1]")
        return value

    @property
    def rho_min(self) -> float:
        """Return the caller-supplied rho lower bound."""

        return self._rho_min

    @staticmethod
    def _validate_attention_mask(
        text_embeddings: Tensor,
        text_attention_mask: Optional[Tensor],
    ) -> None:
        if text_attention_mask is None:
            return
        if not isinstance(text_attention_mask, Tensor):
            raise TypeError("text_attention_mask must be a torch.Tensor or None")
        expected_shape = text_embeddings.shape[:2]
        if text_attention_mask.ndim != 2 or text_attention_mask.shape != expected_shape:
            raise ValueError(
                "text_attention_mask must have shape "
                f"{tuple(expected_shape)}, got {tuple(text_attention_mask.shape)}"
            )
        if text_attention_mask.device != text_embeddings.device:
            raise ValueError(
                "text_attention_mask and text_embeddings must be on the same device"
            )

        if text_attention_mask.dtype == torch.bool:
            valid_mask = text_attention_mask
        else:
            if text_attention_mask.is_complex():
                raise TypeError("text_attention_mask must be boolean or numeric zero/one")
            numeric_dtypes = {
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            }
            if not (
                torch.is_floating_point(text_attention_mask)
                or text_attention_mask.dtype in numeric_dtypes
            ):
                raise TypeError("text_attention_mask must be boolean or numeric zero/one")
            if not bool(torch.isfinite(text_attention_mask).all().item()):
                raise ValueError("text_attention_mask must contain only finite values")
            is_binary = (text_attention_mask == 0) | (text_attention_mask == 1)
            if not bool(is_binary.all().item()):
                raise ValueError("numeric text_attention_mask values must be zero or one")
            valid_mask = text_attention_mask == 1

        if not bool(valid_mask.any(dim=1).all().item()):
            raise ValueError("each batch item must contain at least one valid text token")

    @staticmethod
    def _validated_representation(
        module_name: str,
        representation: Tensor,
        batch_size: int,
        expected_dim: int,
    ) -> Tensor:
        if not isinstance(representation, Tensor):
            raise TypeError(f"{module_name} must return a torch.Tensor")
        expected_shape = (batch_size, expected_dim)
        if representation.ndim != 2 or representation.shape != expected_shape:
            raise ValueError(
                f"{module_name} must return shape {expected_shape}, "
                f"got {tuple(representation.shape)}"
            )
        if not torch.is_floating_point(representation):
            raise TypeError(f"{module_name} output must have a floating-point dtype")
        if not bool(torch.isfinite(representation).all().item()):
            raise ValueError(f"{module_name} output must contain only finite values")
        return representation

    @staticmethod
    def _validated_control(
        head_name: str,
        values: Tensor,
        batch_size: int,
        lower_bound: float,
    ) -> Tensor:
        if not isinstance(values, Tensor):
            raise TypeError(f"{head_name} must return a torch.Tensor")
        expected_shape = (batch_size,)
        if values.ndim != 1 or values.shape != expected_shape:
            raise ValueError(
                f"{head_name} must return shape {expected_shape}, "
                f"got {tuple(values.shape)}"
            )
        if not torch.is_floating_point(values):
            raise TypeError(f"{head_name} output must have a floating-point dtype")
        if not bool(torch.isfinite(values).all().item()):
            raise ValueError(f"{head_name} output must contain only finite values")
        in_range = (lower_bound <= values) & (values <= 1.0)
        if not bool(in_range.all().item()):
            raise ValueError(
                f"{head_name} output must lie in [{lower_bound}, 1]"
            )
        return values

    def forward(
        self,
        text_embeddings: Tensor,
        text_attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Return validated final lambda and rho values from injected heads."""

        if not isinstance(text_embeddings, Tensor):
            raise TypeError("text_embeddings must be a torch.Tensor")
        if text_embeddings.ndim != 3:
            raise ValueError(
                "text_embeddings must have shape "
                "[batch, text_tokens, host_embedding_dim]"
            )
        batch_size, text_tokens, host_embedding_dim = text_embeddings.shape
        if batch_size <= 0:
            raise ValueError("text_embeddings batch dimension must be non-empty")
        if text_tokens <= 0:
            raise ValueError("text_embeddings must contain at least one text token")
        if host_embedding_dim <= 0:
            raise ValueError("text_embeddings host embedding dimension must be non-empty")
        if not torch.is_floating_point(text_embeddings):
            raise TypeError("text_embeddings must have a floating-point dtype")
        if not bool(torch.isfinite(text_embeddings).all().item()):
            raise ValueError("text_embeddings must contain only finite values")
        self._validate_attention_mask(text_embeddings, text_attention_mask)

        instruction = self.instruction_encoder(
            text_embeddings, text_attention_mask
        )
        instruction = self._validated_representation(
            "instruction_encoder", instruction, batch_size, self.instruction_dim
        )
        shared_features = self.shared_mlp(instruction)
        shared_features = self._validated_representation(
            "shared_mlp", shared_features, batch_size, self.shared_dim
        )

        lambda_value = self._validated_control(
            "lambda_head", self.lambda_head(shared_features), batch_size, 0.0
        )
        rho = self._validated_control(
            "rho_head", self.rho_head(shared_features), batch_size, self.rho_min
        )
        return lambda_value, rho


__all__ = ["DualityAwareRouter"]
