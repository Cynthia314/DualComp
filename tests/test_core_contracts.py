"""CPU-only tests for paper equations and public stateless primitives."""

import inspect
import math
import unittest

import dualcomp
import torch

from dualcomp.attention import cls_to_patch_attention
from dualcomp.compressor import (
    ContinuousTokenBudget,
    TokenBudget,
    allocate_token_budget,
    continuous_token_budget,
)
from dualcomp.fusion import lambda_weighted_concatenation


def exact_integer_policy(continuous: ContinuousTokenBudget) -> TokenBudget:
    """Accept only a test case whose paper equations already yield integers."""

    values = (continuous.keep, continuous.semantic, continuous.geometric)
    if not all(value.is_integer() for value in values):
        raise ValueError("the continuous allocation is not already integral")
    return TokenBudget(*(int(value) for value in values))


class FusionTests(unittest.TestCase):
    def test_exact_weighted_concatenation_and_order(self):
        semantic = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        geometric = torch.tensor([[10.0, 20.0]])
        result = lambda_weighted_concatenation(semantic, geometric, 0.25)
        expected = torch.tensor([[0.75, 1.5], [2.25, 3.0], [2.5, 5.0]])
        torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_batch_duality_is_applied_per_sample(self):
        semantic = torch.ones(2, 1, 1)
        geometric = torch.full((2, 1, 1), 2.0)
        duality = torch.tensor([0.0, 1.0])
        result = lambda_weighted_concatenation(semantic, geometric, duality)
        expected = torch.tensor([[[1.0], [0.0]], [[0.0], [2.0]]])
        torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_fusion_rejects_ambiguous_duality_broadcast(self):
        with self.assertRaises(ValueError):
            lambda_weighted_concatenation(
                torch.ones(2, 1, 3),
                torch.ones(2, 1, 3),
                torch.ones(2, 1),
            )


class AttentionTests(unittest.TestCase):
    def test_equation_two_matches_manual_scaled_dot_product(self):
        query = torch.tensor(
            [[[[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]],
              [[0.0, 1.0], [2.0, 0.0], [1.0, -1.0]]]]
        )
        key = torch.tensor(
            [[[[1.0, 1.0], [9.0, 9.0], [0.0, 2.0]],
              [[1.0, 0.0], [8.0, 8.0], [-1.0, 1.0]]]]
        )
        actual = cls_to_patch_attention(
            query,
            key,
            cls_query_index=1,
            patch_key_indices=[0, 2],
        )
        cls = query[..., 1, :]
        patches = key.index_select(-2, torch.tensor([0, 2]))
        logits = torch.matmul(cls.unsqueeze(-2), patches.transpose(-1, -2)).squeeze(-2)
        expected = torch.softmax(logits / math.sqrt(2), dim=-1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual.sum(-1), torch.ones(1, 2), rtol=0, atol=0)

    def test_indices_are_required_and_validated(self):
        query = torch.ones(3, 2)
        key = torch.ones(3, 2)
        with self.assertRaises(TypeError):
            cls_to_patch_attention(query, key, cls_query_index=0)
        with self.assertRaises(ValueError):
            cls_to_patch_attention(
                query,
                key,
                cls_query_index=0,
                patch_key_indices=[1, 1],
            )

    def test_no_head_reduction_is_hidden(self):
        query = torch.randn(2, 3, 4, 5)
        key = torch.randn(2, 3, 6, 5)
        result = cls_to_patch_attention(
            query,
            key,
            cls_query_index=0,
            patch_key_indices=[1, 2, 4],
        )
        self.assertEqual(result.shape, (2, 3, 3))


class BudgetTests(unittest.TestCase):
    def test_continuous_budget_matches_paper_equations_without_rounding(self):
        budget = continuous_token_budget(
            max_tokens=11,
            rho=0.8,
            duality=0.3,
            rho_min=0.1,
        )
        self.assertAlmostEqual(budget.keep, 8.8)
        self.assertAlmostEqual(budget.semantic, 6.16)
        self.assertAlmostEqual(budget.geometric, 2.64)

    def test_continuous_budget_rejects_batched_router_outputs(self):
        with self.assertRaisesRegex(ValueError, "exactly one value"):
            continuous_token_budget(
                max_tokens=10,
                rho=torch.tensor([0.8, 0.9]),
                duality=0.25,
                rho_min=0.1,
            )
        with self.assertRaisesRegex(ValueError, "exactly one value"):
            continuous_token_budget(
                max_tokens=10,
                rho=0.8,
                duality=torch.tensor([0.25, 0.75]),
                rho_min=0.1,
            )

    def test_discrete_allocation_requires_policy(self):
        parameter = inspect.signature(allocate_token_budget).parameters["budget_policy"]
        self.assertIs(parameter.default, inspect.Parameter.empty)
        with self.assertRaises(TypeError):
            allocate_token_budget(max_tokens=11, rho=0.8, duality=0.3, rho_min=0.1)

    def test_nonintegral_allocation_has_no_hidden_rounding_policy(self):
        with self.assertRaisesRegex(ValueError, "not already integral"):
            allocate_token_budget(
                max_tokens=11,
                rho=0.8,
                duality=0.3,
                rho_min=0.1,
                budget_policy=exact_integer_policy,
            )

    def test_explicit_exact_integer_policy_succeeds(self):
        budget = allocate_token_budget(
            max_tokens=10,
            rho=0.8,
            duality=0.25,
            rho_min=0.1,
            budget_policy=exact_integer_policy,
        )
        self.assertEqual(budget, TokenBudget(keep=8, semantic=6, geometric=2))

    def test_zero_token_stream_is_allowed_when_policy_selects_it(self):
        budget = allocate_token_budget(
            max_tokens=10,
            rho=0.8,
            duality=0.0,
            rho_min=0.1,
            budget_policy=lambda _continuous: TokenBudget(
                keep=8,
                semantic=8,
                geometric=0,
            ),
        )
        self.assertEqual(budget, TokenBudget(keep=8, semantic=8, geometric=0))

    def test_invalid_discrete_budgets_are_rejected(self):
        invalid = (
            TokenBudget(keep=-1, semantic=0, geometric=0),
            TokenBudget(keep=3, semantic=1, geometric=1),
            TokenBudget(keep=12, semantic=6, geometric=6),
            TokenBudget(keep=3.0, semantic=1, geometric=2),
        )
        for budget in invalid:
            with self.subTest(budget=budget), self.assertRaises((TypeError, ValueError)):
                allocate_token_budget(
                    max_tokens=11,
                    rho=0.8,
                    duality=0.3,
                    rho_min=0.1,
                    budget_policy=lambda _continuous, result=budget: result,
                )

    def test_invalid_policy_result_type_is_rejected(self):
        with self.assertRaises(TypeError):
            allocate_token_budget(
                max_tokens=11,
                rho=0.8,
                duality=0.3,
                rho_min=0.1,
                budget_policy=lambda _continuous: (8, 5, 3),
            )


class PublicAPITests(unittest.TestCase):
    def test_public_exports_match_reviewed_contract(self):
        expected = {
            "BudgetPolicy",
            "ContinuousTokenBudget",
            "DualityAwareRouter",
            "IGSRResult",
            "ParameterFreeIGSR",
            "SCSAResult",
            "SpatiallyContiguousSemanticAggregator",
            "TokenBudget",
            "allocate_token_budget",
            "cls_to_patch_attention",
            "continuous_token_budget",
            "lambda_weighted_concatenation",
            "local_difference_saliency",
            "recover_structure",
            "select_region_anchors",
            "spatially_contiguous_semantic_aggregate",
            "structural_modulation",
            "trace_greedy_path",
        }
        self.assertEqual(set(dualcomp.__all__), expected)
        for name in expected:
            with self.subTest(name=name):
                self.assertTrue(hasattr(dualcomp, name))


if __name__ == "__main__":
    unittest.main()
