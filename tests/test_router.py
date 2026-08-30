"""CPU tests for the strictly paper-bounded router wrapper."""

from __future__ import annotations

import inspect
import unittest

import torch
from torch import nn

from dualcomp.router import DualityAwareRouter


class StubInstructionEncoder(nn.Module):
    def __init__(self, host_dim: int, instruction_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(host_dim, instruction_dim)
        self.last_embeddings = None
        self.last_attention_mask = None
        self.last_output = None

    def forward(self, text_embeddings, text_attention_mask):
        self.last_embeddings = text_embeddings
        self.last_attention_mask = text_attention_mask
        self.last_output = self.projection(text_embeddings[:, 0, :])
        return self.last_output


class StubSharedMLP(nn.Module):
    def __init__(self, instruction_dim: int, shared_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(instruction_dim, shared_dim)
        self.last_input = None

    def forward(self, instruction):
        self.last_input = instruction
        return self.projection(instruction)


class ExplicitBoundedHead(nn.Module):
    """A caller-side test choice, not a router default implementation."""

    def __init__(self, shared_dim: int, lower_bound: float, upper_bound: float) -> None:
        super().__init__()
        self.projection = nn.Linear(shared_dim, 1)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def forward(self, shared_features):
        unit_value = torch.sigmoid(self.projection(shared_features)).squeeze(-1)
        return self.lower_bound + (self.upper_bound - self.lower_bound) * unit_value


class FixedInstructionEncoder(nn.Module):
    def __init__(self, output) -> None:
        super().__init__()
        self.output = output

    def forward(self, text_embeddings, text_attention_mask):
        return self.output


class FixedUnaryModule(nn.Module):
    def __init__(self, output) -> None:
        super().__init__()
        self.output = output

    def forward(self, inputs):
        return self.output


class DualityAwareRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.rho_min = 0.2
        self.encoder = StubInstructionEncoder(host_dim=4, instruction_dim=3)
        self.shared = StubSharedMLP(instruction_dim=3, shared_dim=2)
        self.lambda_head = ExplicitBoundedHead(
            shared_dim=2, lower_bound=0.0, upper_bound=1.0
        )
        self.rho_head = ExplicitBoundedHead(
            shared_dim=2, lower_bound=self.rho_min, upper_bound=1.0
        )
        self.router = DualityAwareRouter(
            instruction_encoder=self.encoder,
            shared_mlp=self.shared,
            lambda_head=self.lambda_head,
            rho_head=self.rho_head,
            instruction_dim=3,
            shared_dim=2,
            rho_min=self.rho_min,
        ).cpu()
        self.text_embeddings = torch.tensor(
            [
                [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
                [[-0.1, -0.2, -0.3, -0.4], [-0.5, -0.6, -0.7, -0.8]],
            ],
            dtype=torch.float32,
        )

    def test_constructor_has_no_component_or_dimension_defaults(self) -> None:
        signature = inspect.signature(DualityAwareRouter.__init__)
        required = (
            "instruction_encoder",
            "shared_mlp",
            "lambda_head",
            "rho_head",
            "instruction_dim",
            "shared_dim",
            "rho_min",
        )
        for name in required:
            self.assertIs(signature.parameters[name].default, inspect.Parameter.empty)

        with self.assertRaises(TypeError):
            DualityAwareRouter(
                self.encoder,
                self.shared,
                self.lambda_head,
                self.rho_head,
                3,
                2,
            )

    def test_wrapper_has_no_direct_trainable_parameters_or_buffers(self) -> None:
        self.assertEqual(list(self.router.named_parameters(recurse=False)), [])
        self.assertEqual(list(self.router.named_buffers(recurse=False)), [])
        self.assertGreater(len(list(self.router.parameters())), 0)

    def test_injected_modules_are_used_without_internal_substitutes(self) -> None:
        self.assertIs(self.router.instruction_encoder, self.encoder)
        self.assertIs(self.router.shared_mlp, self.shared)
        self.assertIs(self.router.lambda_head, self.lambda_head)
        self.assertIs(self.router.rho_head, self.rho_head)
        self.assertFalse(hasattr(self.router, "adapter"))
        self.assertFalse(hasattr(self.router, "query"))

    def test_forward_shapes_and_ranges(self) -> None:
        lambda_value, rho = self.router(self.text_embeddings)

        self.assertEqual(tuple(lambda_value.shape), (2,))
        self.assertEqual(tuple(rho.shape), (2,))
        self.assertTrue(bool(((0.0 <= lambda_value) & (lambda_value <= 1.0)).all()))
        self.assertTrue(bool(((self.rho_min <= rho) & (rho <= 1.0)).all()))

    def test_head_outputs_are_returned_without_wrapper_mapping(self) -> None:
        expected_lambda = torch.tensor([0.0, 1.0])
        expected_rho = torch.tensor([self.rho_min, 1.0])
        router = DualityAwareRouter(
            self.encoder,
            self.shared,
            FixedUnaryModule(expected_lambda),
            FixedUnaryModule(expected_rho),
            3,
            2,
            self.rho_min,
        )

        actual_lambda, actual_rho = router(self.text_embeddings)
        self.assertIs(actual_lambda, expected_lambda)
        self.assertIs(actual_rho, expected_rho)

    def test_mask_and_representations_are_passed_unchanged(self) -> None:
        mask = torch.tensor([[True, False], [True, True]])
        self.router(self.text_embeddings, mask)

        self.assertIs(self.encoder.last_embeddings, self.text_embeddings)
        self.assertIs(self.encoder.last_attention_mask, mask)
        self.assertIs(self.shared.last_input, self.encoder.last_output)

        self.router(self.text_embeddings)
        self.assertIsNone(self.encoder.last_attention_mask)

    def test_gradients_flow_through_all_injected_modules(self) -> None:
        embeddings = self.text_embeddings.clone().requires_grad_(True)
        lambda_value, rho = self.router(embeddings)
        (lambda_value.sum() + rho.sum()).backward()

        self.assertIsNotNone(embeddings.grad)
        self.assertTrue(bool(torch.isfinite(embeddings.grad).all()))
        for module in (
            self.encoder,
            self.shared,
            self.lambda_head,
            self.rho_head,
        ):
            parameters = list(module.parameters())
            self.assertGreater(len(parameters), 0)
            for parameter in parameters:
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()))

    def test_constructor_rejects_invalid_modules_and_shared_heads(self) -> None:
        valid_arguments = (
            self.encoder,
            self.shared,
            self.lambda_head,
            self.rho_head,
            3,
            2,
            self.rho_min,
        )
        for index in range(4):
            arguments = list(valid_arguments)
            arguments[index] = object()
            with self.subTest(module_index=index):
                with self.assertRaises(TypeError):
                    DualityAwareRouter(*arguments)

        with self.assertRaises(ValueError):
            DualityAwareRouter(
                self.encoder,
                self.shared,
                self.lambda_head,
                self.lambda_head,
                3,
                2,
                self.rho_min,
            )

    def test_constructor_rejects_invalid_dimensions_and_rho_min(self) -> None:
        for name in ("instruction_dim", "shared_dim"):
            for invalid, exception in (
                (0, ValueError),
                (-1, ValueError),
                (True, TypeError),
                (1.5, TypeError),
            ):
                kwargs = dict(
                    instruction_encoder=self.encoder,
                    shared_mlp=self.shared,
                    lambda_head=self.lambda_head,
                    rho_head=self.rho_head,
                    instruction_dim=3,
                    shared_dim=2,
                    rho_min=self.rho_min,
                )
                kwargs[name] = invalid
                with self.subTest(name=name, invalid=invalid):
                    with self.assertRaises(exception):
                        DualityAwareRouter(**kwargs)

        for invalid in (-0.01, 1.01, float("nan"), float("inf")):
            with self.subTest(rho_min=invalid):
                with self.assertRaises(ValueError):
                    DualityAwareRouter(
                        self.encoder,
                        self.shared,
                        self.lambda_head,
                        self.rho_head,
                        3,
                        2,
                        invalid,
                    )

    def test_rejects_invalid_text_embeddings_and_masks(self) -> None:
        invalid_embeddings = (
            torch.zeros(2, 4),
            torch.zeros(2, 0, 4),
            torch.zeros(0, 2, 4),
            torch.zeros(2, 2, 0),
            torch.zeros(2, 2, 4, dtype=torch.int64),
        )
        for invalid in invalid_embeddings:
            with self.subTest(shape=tuple(invalid.shape), dtype=invalid.dtype):
                with self.assertRaises((TypeError, ValueError)):
                    self.router(invalid)

        non_finite = self.text_embeddings.clone()
        non_finite[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            self.router(non_finite)

        invalid_masks = (
            torch.ones(2, 3, dtype=torch.bool),
            torch.tensor([[1, 0], [0, 0]], dtype=torch.int64),
            torch.tensor([[1, 2], [1, 0]], dtype=torch.int64),
            torch.tensor([[1.0, float("nan")], [1.0, 0.0]]),
        )
        for invalid in invalid_masks:
            with self.subTest(mask=invalid):
                with self.assertRaises(ValueError):
                    self.router(self.text_embeddings, invalid)

    def test_validates_instruction_and_shared_shapes(self) -> None:
        invalid_instruction_outputs = (
            None,
            torch.zeros(2, 3, 1),
            torch.zeros(1, 3),
            torch.zeros(2, 4),
            torch.zeros(2, 3, dtype=torch.int64),
            torch.tensor([[float("nan"), 0.0, 0.0], [0.0, 0.0, 0.0]]),
        )
        for invalid in invalid_instruction_outputs:
            router = DualityAwareRouter(
                FixedInstructionEncoder(invalid),
                self.shared,
                self.lambda_head,
                self.rho_head,
                3,
                2,
                self.rho_min,
            )
            with self.subTest(instruction_output=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    router(self.text_embeddings)

        invalid_shared_outputs = (
            None,
            torch.zeros(2, 2, 1),
            torch.zeros(1, 2),
            torch.zeros(2, 3),
            torch.zeros(2, 2, dtype=torch.int64),
            torch.tensor([[float("inf"), 0.0], [0.0, 0.0]]),
        )
        for invalid in invalid_shared_outputs:
            router = DualityAwareRouter(
                self.encoder,
                FixedUnaryModule(invalid),
                self.lambda_head,
                self.rho_head,
                3,
                2,
                self.rho_min,
            )
            with self.subTest(shared_output=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    router(self.text_embeddings)

    def test_validates_lambda_head_shape_finiteness_and_range(self) -> None:
        invalid_outputs = (
            None,
            torch.zeros(2, 1),
            torch.zeros(1),
            torch.zeros(2, dtype=torch.int64),
            torch.tensor([float("nan"), 0.5]),
            torch.tensor([-0.01, 0.5]),
            torch.tensor([0.5, 1.01]),
        )
        for invalid in invalid_outputs:
            router = DualityAwareRouter(
                self.encoder,
                self.shared,
                FixedUnaryModule(invalid),
                self.rho_head,
                3,
                2,
                self.rho_min,
            )
            with self.subTest(lambda_output=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    router(self.text_embeddings)

    def test_validates_rho_head_shape_finiteness_and_range(self) -> None:
        invalid_outputs = (
            None,
            torch.zeros(2, 1),
            torch.zeros(1),
            torch.zeros(2, dtype=torch.int64),
            torch.tensor([float("inf"), 0.5]),
            torch.tensor([self.rho_min - 0.01, 0.5]),
            torch.tensor([0.5, 1.01]),
        )
        for invalid in invalid_outputs:
            router = DualityAwareRouter(
                self.encoder,
                self.shared,
                self.lambda_head,
                FixedUnaryModule(invalid),
                3,
                2,
                self.rho_min,
            )
            with self.subTest(rho_output=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    router(self.text_embeddings)


if __name__ == "__main__":
    unittest.main()
