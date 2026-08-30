"""CPU tests for the clean-room SCSA implementation."""

from __future__ import annotations

import inspect

import pytest
import torch

from dualcomp.scsa import (
    SpatiallyContiguousSemanticAggregator,
    spatially_contiguous_semantic_aggregate,
)


def test_caller_supplied_links_and_cluster_scores_are_cumulative() -> None:
    tokens = torch.tensor(
        [
            [
                [1.00, 0.00, 0.0],
                [0.99, 0.10, 0.0],
                [0.00, 1.00, 0.0],
                [1.00, 0.00, 0.0],
            ]
        ]
    )
    attention = torch.tensor([[0.20, 0.25, 0.40, 0.15]])

    result = spatially_contiguous_semantic_aggregate(
        tokens,
        attention,
        height=1,
        width=4,
        budget=1,
        tau=0.99,
        size_threshold=2,
        predecessor_neighborhoods=[[], [0], [1], [2]],
    )

    # The supplied graph lets patch 1 link to 0, while patch 3 may inspect only 2.
    assert result.token_cluster_ids.tolist() == [[0, 0, 1, 2]]
    # The two-patch cluster wins by attention sum even though patch 2 has the
    # largest individual attention value.
    torch.testing.assert_close(result.importance, torch.tensor([[0.45]]))
    assert result.cluster_sizes.tolist() == [[2]]
    assert result.peak_indices.tolist() == [[1]]
    assert not result.large_cluster_mask.item()
    torch.testing.assert_close(result.compressed_tokens[0, 0], tokens[0, 1])


def test_caller_supplied_neighborhood_controls_connectivity() -> None:
    tokens = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]]
    )
    attention = torch.tensor([[0.40, 0.30, 0.10, 0.05]])

    result = spatially_contiguous_semantic_aggregate(
        tokens,
        attention,
        height=2,
        width=2,
        budget=2,
        tau=0.9,
        size_threshold=4,
        predecessor_neighborhoods=[[], [0], [1], [0, 2]],
    )

    # Only the explicit edge from 3 to 0 permits those matching patches to merge.
    assert result.token_cluster_ids.tolist() == [[0, 1, 1, 0]]
    assert result.cluster_count.tolist() == [2]
    assert result.valid_mask.tolist() == [[True, True]]


def test_large_cluster_uses_attention_weighted_mean() -> None:
    tokens = torch.tensor([[[1.0, 2.0], [2.0, 4.0], [4.0, 8.0]]])
    attention = torch.tensor([[1.0, 2.0, 1.0]])

    result = spatially_contiguous_semantic_aggregate(
        tokens,
        attention,
        height=1,
        width=3,
        budget=1,
        tau=0.99,
        size_threshold=2,
        predecessor_neighborhoods=[[], [0], [1]],
    )

    expected = (tokens[0, 0] + 2 * tokens[0, 1] + tokens[0, 2]) / 4
    torch.testing.assert_close(result.compressed_tokens[0, 0], expected)
    assert result.large_cluster_mask.item()
    assert result.cluster_sizes.item() == 3
    assert result.peak_indices.item() == -1


def test_zero_attention_large_cluster_raises_without_paper_fallback() -> None:
    tokens = torch.tensor([[[1.0, 0.0], [3.0, 0.0]]])
    attention = torch.zeros((1, 2))

    with pytest.raises(ValueError, match="zero total CLS attention"):
        spatially_contiguous_semantic_aggregate(
            tokens,
            attention,
            height=1,
            width=2,
            budget=1,
            tau=0.9,
            size_threshold=0,
            predecessor_neighborhoods=[[], [0]],
        )


def test_zero_norm_token_is_rejected_instead_of_defining_cosine() -> None:
    tokens = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])
    attention = torch.tensor([[0.6, 0.4]])

    with pytest.raises(ValueError, match="cosine similarity is undefined"):
        spatially_contiguous_semantic_aggregate(
            tokens,
            attention,
            height=1,
            width=2,
            budget=1,
            tau=0.9,
            size_threshold=1,
            predecessor_neighborhoods=[[], [0]],
        )


def test_isolated_zero_norm_token_does_not_require_a_cosine_convention() -> None:
    tokens = torch.tensor([[[0.0, 0.0]]])
    attention = torch.tensor([[1.0]])

    result = spatially_contiguous_semantic_aggregate(
        tokens,
        attention,
        height=1,
        width=1,
        budget=1,
        tau=0.9,
        size_threshold=1,
        predecessor_neighborhoods=[[]],
    )

    torch.testing.assert_close(result.compressed_tokens, tokens)


def test_cosine_argmax_tie_is_rejected_without_hidden_predecessor_order() -> None:
    tokens = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])
    attention = torch.tensor([[0.5, 0.3, 0.2]])

    with pytest.raises(ValueError, match="cosine argmax is tied"):
        spatially_contiguous_semantic_aggregate(
            tokens,
            attention,
            height=1,
            width=3,
            budget=1,
            tau=0.9,
            size_threshold=1,
            predecessor_neighborhoods=[[], [], [0, 1]],
        )


def test_cluster_importance_tie_is_rejected_at_budget_boundary() -> None:
    tokens = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    attention = torch.tensor([[0.5, 0.5]])

    with pytest.raises(ValueError, match="cluster-importance tie"):
        spatially_contiguous_semantic_aggregate(
            tokens,
            attention,
            height=1,
            width=2,
            budget=1,
            tau=0.9,
            size_threshold=1,
            predecessor_neighborhoods=[[], [0]],
        )


def test_small_cluster_peak_tie_is_rejected() -> None:
    tokens = torch.tensor([[[1.0, 0.0], [2.0, 0.0]]])
    attention = torch.tensor([[0.5, 0.5]])

    with pytest.raises(ValueError, match="peak attention is tied"):
        spatially_contiguous_semantic_aggregate(
            tokens,
            attention,
            height=1,
            width=2,
            budget=1,
            tau=0.9,
            size_threshold=2,
            predecessor_neighborhoods=[[], [0]],
        )


def test_batch_controls_can_be_injected_without_padding() -> None:
    tokens = torch.tensor(
        [
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
        ]
    )
    attention = torch.tensor([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]])
    tau_calls = 0
    size_calls = 0

    def tau_control(received_tokens: torch.Tensor, received_attention: torch.Tensor):
        nonlocal tau_calls
        tau_calls += 1
        assert received_tokens is tokens
        assert received_attention is attention
        return torch.tensor([0.9, 0.9])

    def size_control(received_tokens: torch.Tensor, received_attention: torch.Tensor):
        nonlocal size_calls
        size_calls += 1
        assert received_tokens is tokens
        assert received_attention is attention
        return torch.tensor([1.0, 2.0])

    result = spatially_contiguous_semantic_aggregate(
        tokens,
        attention,
        height=1,
        width=3,
        budget=1,
        tau=tau_control,
        size_threshold=size_control,
        predecessor_neighborhoods=[[], [0], [1]],
    )

    assert tau_calls == 1
    assert size_calls == 1
    assert result.compressed_tokens.shape == (2, 1, 2)
    assert result.cluster_count.tolist() == [1, 3]
    assert result.valid_mask.tolist() == [[True], [True]]
    assert result.peak_indices.tolist() == [[-1], [0]]


def test_budget_larger_than_formed_cluster_count_is_rejected_without_padding() -> None:
    tokens = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]])
    attention = torch.tensor([[0.2, 0.3, 0.5]])

    with pytest.raises(ValueError, match="paper does not define padding"):
        spatially_contiguous_semantic_aggregate(
            tokens,
            attention,
            height=1,
            width=3,
            budget=2,
            tau=0.9,
            size_threshold=1,
            predecessor_neighborhoods=[[], [0], [1]],
        )


def test_threshold_comparison_is_strict() -> None:
    tokens = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    attention = torch.tensor([[0.6, 0.4]])

    result = spatially_contiguous_semantic_aggregate(
        tokens,
        attention,
        height=1,
        width=2,
        budget=2,
        tau=1.0,
        size_threshold=1,
        predecessor_neighborhoods=[[], [0]],
    )

    assert result.cluster_count.tolist() == [2]
    assert result.token_cluster_ids.tolist() == [[0, 1]]


def test_module_is_parameter_free_and_preserves_weighted_mean_gradients() -> None:
    module = SpatiallyContiguousSemanticAggregator()
    assert list(module.parameters()) == []

    tokens = torch.tensor(
        [[[1.0, 0.0], [2.0, 0.0]]], requires_grad=True
    )
    attention = torch.tensor([[1.0, 3.0]])
    result = module(
        tokens,
        attention,
        height=1,
        width=2,
        budget=1,
        tau=0.9,
        size_threshold=0,
        predecessor_neighborhoods=[[], [0]],
    )
    result.compressed_tokens.sum().backward()

    expected_gradient = torch.tensor([[[0.25, 0.25], [0.75, 0.75]]])
    torch.testing.assert_close(tokens.grad, expected_gradient)


def test_output_dtype_and_device_follow_tokens_without_downcasting_attention() -> None:
    tokens = torch.tensor([[[1.0, 0.0], [3.0, 0.0]]], dtype=torch.float32)
    attention = torch.tensor([[1.0, 3.0]], dtype=torch.float64)

    result = spatially_contiguous_semantic_aggregate(
        tokens,
        attention,
        height=1,
        width=2,
        budget=1,
        tau=0.9,
        size_threshold=0,
        predecessor_neighborhoods=[[], [0]],
    )

    assert result.compressed_tokens.dtype == tokens.dtype
    assert result.compressed_tokens.device == tokens.device
    assert result.importance.dtype == attention.dtype
    torch.testing.assert_close(
        result.compressed_tokens,
        torch.tensor([[[2.5, 0.0]]], dtype=torch.float32),
    )


def test_control_tensor_is_not_silently_moved_between_devices() -> None:
    tokens = torch.tensor([[[1.0, 0.0]]])
    attention = torch.tensor([[1.0]])
    tau_on_meta_device = torch.empty((), device="meta")

    with pytest.raises(ValueError, match="same device as tokens"):
        spatially_contiguous_semantic_aggregate(
            tokens,
            attention,
            height=1,
            width=1,
            budget=1,
            tau=tau_on_meta_device,
            size_threshold=1,
            predecessor_neighborhoods=[[]],
        )


def test_controls_are_required_and_invalid_inputs_are_rejected() -> None:
    signature = inspect.signature(spatially_contiguous_semantic_aggregate)
    assert signature.parameters["tau"].default is inspect.Parameter.empty
    assert signature.parameters["size_threshold"].default is inspect.Parameter.empty
    assert (
        signature.parameters["predecessor_neighborhoods"].default
        is inspect.Parameter.empty
    )

    tokens = torch.ones((1, 4, 2))
    attention = torch.ones((1, 4))

    with pytest.raises(ValueError, match="budget must not exceed N"):
        spatially_contiguous_semantic_aggregate(
            tokens,
            attention,
            2,
            2,
            5,
            tau=0.5,
            size_threshold=1,
            predecessor_neighborhoods=[[], [0], [1], [2]],
        )
    with pytest.raises(ValueError, match=r"height \* width"):
        spatially_contiguous_semantic_aggregate(
            tokens,
            attention,
            1,
            3,
            1,
            tau=0.5,
            size_threshold=1,
            predecessor_neighborhoods=[[], [0], [1], [2]],
        )
    with pytest.raises(ValueError, match="non-negative attention"):
        spatially_contiguous_semantic_aggregate(
            tokens,
            torch.tensor([[1.0, -1.0, 1.0, 1.0]]),
            2,
            2,
            1,
            tau=0.5,
            size_threshold=1,
            predecessor_neighborhoods=[[], [0], [1], [2]],
        )
    with pytest.raises(ValueError, match="size_threshold must contain non-negative"):
        spatially_contiguous_semantic_aggregate(
            tokens,
            attention,
            2,
            2,
            1,
            tau=0.5,
            size_threshold=-1,
            predecessor_neighborhoods=[[], [0], [1], [2]],
        )
    with pytest.raises(ValueError, match=r"shape \[B\]"):
        spatially_contiguous_semantic_aggregate(
            tokens,
            attention,
            2,
            2,
            1,
            tau=torch.tensor([0.1, 0.2]),
            size_threshold=1,
            predecessor_neighborhoods=[[], [0], [1], [2]],
        )


@pytest.mark.parametrize(
    ("neighborhoods", "message"),
    [
        ([[], [0], [1]], "length N"),
        ([[0], [], [], []], "0 <= j < i"),
        ([[], [0, 0], [], []], "unique"),
        ([[], ["0"], [], []], "must be integers"),
        ([[], [1], [], []], "0 <= j < i"),
    ],
)
def test_invalid_predecessor_neighborhoods_are_rejected(
    neighborhoods, message: str
) -> None:
    tokens = torch.ones((1, 4, 2))
    attention = torch.ones((1, 4))

    with pytest.raises((TypeError, ValueError), match=message):
        spatially_contiguous_semantic_aggregate(
            tokens,
            attention,
            height=2,
            width=2,
            budget=1,
            tau=0.5,
            size_threshold=1,
            predecessor_neighborhoods=neighborhoods,
        )


def test_zero_budget_returns_well_shaped_empty_selection() -> None:
    tokens = torch.tensor([[[1.0], [1.0]]])
    attention = torch.tensor([[0.4, 0.6]])

    result = spatially_contiguous_semantic_aggregate(
        tokens,
        attention,
        height=1,
        width=2,
        budget=0,
        tau=0.9,
        size_threshold=1,
        predecessor_neighborhoods=[[], [0]],
    )

    assert result.compressed_tokens.shape == (1, 0, 1)
    assert result.valid_mask.shape == (1, 0)
    assert result.cluster_count.tolist() == [1]
