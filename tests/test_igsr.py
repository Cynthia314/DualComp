import inspect
import unittest

import torch

from dualcomp.igsr import (
    ParameterFreeIGSR,
    local_difference_saliency,
    recover_structure,
    select_region_anchors,
    structural_modulation,
    trace_greedy_path,
)


CALLER_OFFSETS = torch.tensor(
    [
        [-1, -1],
        [-1, 0],
        [-1, 1],
        [0, -1],
        [0, 1],
        [1, -1],
        [1, 0],
        [1, 1],
    ],
    dtype=torch.long,
)


class LocalDifferenceTests(unittest.TestCase):
    def test_squared_l2_difference_uses_caller_local_average(self):
        features = torch.tensor(
            [
                [
                    [[3.0, 1.0], [0.0, -1.0]],
                    [[2.0, 0.0], [1.0, 3.0]],
                ]
            ]
        )
        local_average = torch.tensor(
            [
                [
                    [[1.0, 1.0], [1.0, -1.0]],
                    [[0.0, 1.0], [1.0, 1.0]],
                ]
            ]
        )
        expected = torch.tensor([[[8.0, 1.0], [1.0, 4.0]]])
        torch.testing.assert_close(
            local_difference_saliency(features, local_average), expected
        )

    def test_local_average_requires_finite_floating_same_shape_and_device(self):
        signature = inspect.signature(local_difference_saliency)
        self.assertEqual(
            signature.parameters["local_average"].default, inspect.Parameter.empty
        )
        features = torch.zeros((1, 2, 3, 3))
        with self.assertRaisesRegex(ValueError, "same shape"):
            local_difference_saliency(features, torch.zeros((1, 2, 2, 3)))
        with self.assertRaisesRegex(ValueError, "finite"):
            invalid = torch.zeros_like(features)
            invalid[0, 0, 0, 0] = float("nan")
            local_difference_saliency(features, invalid)
        mixed_dtype = local_difference_saliency(
            features, torch.zeros_like(features).double()
        )
        self.assertTrue(mixed_dtype.is_floating_point())


class ModulationTests(unittest.TestCase):
    def test_tasm_applies_caller_scores_and_explicit_beta(self):
        geometric = torch.tensor([[[-2.0, 0.5], [1.5, 3.0]]])
        text = torch.tensor([[[2.0, -1.0], [0.25, 0.75]]])
        expected = torch.tensor([[[6.0, 1.5], [0.75, -1.5]]])
        torch.testing.assert_close(
            structural_modulation(geometric, text, beta=-2.0), expected
        )

    def test_no_unstated_score_range_or_beta_sign_constraint(self):
        geometric = torch.tensor([[[-1.0, 2.0]]])
        text = torch.tensor([[[3.0, -4.0]]], dtype=torch.float64)
        result = structural_modulation(geometric, text, beta=-0.5)
        expected = torch.tensor([[[0.5, 6.0]]], dtype=torch.float64)
        torch.testing.assert_close(result, expected)

    def test_score_maps_and_beta_require_finite_explicit_inputs(self):
        signature = inspect.signature(structural_modulation)
        self.assertEqual(
            signature.parameters["beta"].default, inspect.Parameter.empty
        )
        valid = torch.tensor([[[0.0, 1.0]]])
        with self.assertRaisesRegex(ValueError, "finite"):
            structural_modulation(
                torch.tensor([[[0.0, float("nan")]]]), valid, beta=1.0
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            structural_modulation(
                valid, torch.tensor([[[0.0, float("inf")]]]), beta=1.0
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            structural_modulation(valid, valid, beta=float("inf"))
        with self.assertRaisesRegex(TypeError, "real scalar"):
            structural_modulation(valid, valid, beta=True)

    def test_score_shapes_and_devices_must_match(self):
        geometric = torch.zeros((1, 2, 2))
        text = torch.zeros((1, 2, 1))
        with self.assertRaisesRegex(ValueError, "must match"):
            structural_modulation(geometric, text, beta=1.0)


class AnchorAndPathTests(unittest.TestCase):
    def test_arbitrary_integer_labels_explicitly_define_anchor_order(self):
        geometric = torch.tensor(
            [[[9.0, 8.0, 1.0], [2.0, 3.0, 5.0], [7.0, 6.0, 4.0]]]
        )
        regions = torch.tensor(
            [[10, 10, -3], [10, 10, -3], [7, 7, 42]], dtype=torch.long
        )
        expected = torch.tensor([[[1, 2], [2, 0], [0, 0], [2, 2]]])
        self.assertTrue(torch.equal(select_region_anchors(geometric, regions), expected))

    def test_region_grid_requires_integer_labels(self):
        geometric = torch.zeros((1, 2, 2))
        with self.assertRaisesRegex(TypeError, "integer dtype"):
            select_region_anchors(
                geometric, torch.tensor([[0.0, 0.0], [1.0, 1.0]])
            )
        with self.assertRaisesRegex(TypeError, "integer dtype"):
            select_region_anchors(
                geometric, torch.tensor([[False, False], [True, True]])
            )

    def test_tied_anchor_maximum_fails_closed_without_hidden_tie_rule(self):
        geometric = torch.tensor([[[5.0, 5.0], [1.0, 0.0]]])
        regions = torch.zeros((2, 2), dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "unique.*maximum"):
            select_region_anchors(geometric, regions)

    def test_greedy_path_uses_explicit_offsets_and_strict_chebyshev_progress(self):
        scores = torch.zeros((3, 3))
        scores[1, 0] = 9.0
        scores[1, 1] = 3.0
        path = trace_greedy_path(scores, (0, 0), (2, 1), CALLER_OFFSETS)
        expected = torch.tensor([[0, 0], [1, 0], [2, 1]])
        self.assertTrue(torch.equal(path, expected))
        goal = expected[-1]
        distances = (path - goal).abs().amax(dim=1)
        self.assertTrue(bool((distances[1:] < distances[:-1]).all().item()))

    def test_caller_can_supply_non_unit_neighborhood_offsets(self):
        offsets = torch.tensor([[0, 2], [0, -2]], dtype=torch.long)
        path = trace_greedy_path(torch.zeros((1, 5)), (0, 0), (0, 4), offsets)
        self.assertTrue(
            torch.equal(path, torch.tensor([[0, 0], [0, 2], [0, 4]]))
        )

    def test_score_ties_follow_only_caller_offset_order(self):
        offsets = torch.tensor([[1, 1], [1, 0], [0, 1]], dtype=torch.long)
        path = trace_greedy_path(torch.zeros((3, 3)), (0, 0), (2, 1), offsets)
        self.assertTrue(torch.equal(path[1], torch.tensor([1, 1])))

    def test_neighbor_offsets_are_required_unique_nonzero_integers(self):
        signature = inspect.signature(trace_greedy_path)
        self.assertEqual(
            signature.parameters["neighbor_offsets"].default,
            inspect.Parameter.empty,
        )
        scores = torch.zeros((3, 3))
        invalid_cases = (
            (torch.empty((0, 2), dtype=torch.long), "at least one"),
            (torch.tensor([[0, 0]], dtype=torch.long), "zero offset"),
            (torch.tensor([[1, 0], [1, 0]], dtype=torch.long), "unique"),
            (torch.tensor([[1.0, 0.0]]), "integer dtype"),
        )
        for offsets, message in invalid_cases:
            with self.subTest(offsets=offsets, message=message):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    trace_greedy_path(scores, (0, 0), (2, 1), offsets)


class EndToEndTests(unittest.TestCase):
    @staticmethod
    def fixture():
        features = torch.zeros((1, 1, 4, 4))
        features[0, 0, 0, 0] = 10.0
        features[0, 0, 0, 3] = 10.0
        features[0, 0, 3, 3] = 10.0
        features[0, 0, 3, 0] = 10.0
        local_average = torch.zeros_like(features)
        normalized_geometric = (features[:, 0] / 10.0).square()
        text = torch.zeros((1, 4, 4))
        regions = torch.tensor(
            [
                [10, 10, 20, 20],
                [10, 10, 20, 20],
                [40, 40, 30, 30],
                [40, 40, 30, 30],
            ],
            dtype=torch.long,
        )
        return (
            features,
            local_average,
            normalized_geometric,
            text,
            regions,
            CALLER_OFFSETS,
        )

    def test_module_is_parameter_free_and_preserves_reference_sequence_contract(self):
        args = self.fixture()
        features, _, _, _, _, offsets = args
        module = ParameterFreeIGSR()
        self.assertEqual(list(module.parameters()), [])

        result = module(*args, beta=0.0, budget=12)
        expected_anchors = torch.tensor([[[0, 0], [0, 3], [3, 3], [3, 0]]])
        self.assertTrue(torch.equal(result.anchors, expected_anchors))
        self.assertEqual(result.sequence_lengths.tolist(), [10])
        self.assertEqual(result.tokens.shape, (1, 12, 1))
        self.assertEqual(result.valid_mask.sum().item(), 10)

        route = result.coordinates[0, result.valid_mask[0]]
        self.assertTrue(
            torch.equal(route[[0, 3, 6, 9]], expected_anchors.squeeze(0))
        )
        deltas = route[1:] - route[:-1]
        for delta in deltas:
            self.assertTrue(bool((offsets == delta).all(dim=1).any().item()))

        gathered = features[0, :, route[:, 0], route[:, 1]].transpose(0, 1)
        torch.testing.assert_close(
            result.tokens[0, result.valid_mask[0]], gathered
        )
        self.assertTrue(bool((result.coordinates[0, 10:] == -1).all().item()))
        torch.testing.assert_close(result.tokens[0, 10:], torch.zeros((2, 1)))

    def test_all_paper_underspecified_inputs_and_budget_are_required(self):
        signature = inspect.signature(recover_structure)
        required = (
            "local_average",
            "normalized_geometric_saliency",
            "normalized_text_relevance",
            "region_grid",
            "neighbor_offsets",
            "beta",
            "budget",
        )
        for parameter in required:
            with self.subTest(parameter=parameter):
                self.assertEqual(
                    signature.parameters[parameter].default,
                    inspect.Parameter.empty,
                )

    def test_budget_overflow_fails_closed_without_truncation(self):
        with self.assertRaisesRegex(ValueError, "at least 10"):
            recover_structure(*self.fixture(), beta=0.0, budget=9)

    def test_budget_has_no_unstated_grid_size_cap(self):
        result = recover_structure(*self.fixture(), beta=0.0, budget=17)
        self.assertEqual(result.tokens.shape, (1, 17, 1))
        self.assertEqual(result.sequence_lengths.tolist(), [10])
        self.assertEqual(result.valid_mask.sum().item(), 10)

    def test_recover_validates_inputs_and_unusable_neighborhood(self):
        args = list(self.fixture())
        with self.assertRaisesRegex(ValueError, "anchor count"):
            recover_structure(*args, beta=0.0, budget=3)

        bad_average = args.copy()
        bad_average[1] = bad_average[1][:, :, :-1]
        with self.assertRaisesRegex(ValueError, "same shape"):
            recover_structure(*bad_average, beta=0.0, budget=12)

        bad_geometric = args.copy()
        bad_geometric[2] = torch.full((1, 4, 4), float("nan"))
        with self.assertRaisesRegex(ValueError, "finite"):
            recover_structure(*bad_geometric, beta=0.0, budget=12)

        unusable_offsets = args.copy()
        unusable_offsets[5] = torch.tensor([[1, 0]], dtype=torch.long)
        with self.assertRaisesRegex(RuntimeError, "supplied neighbor_offsets"):
            recover_structure(*unusable_offsets, beta=0.0, budget=12)


if __name__ == "__main__":
    unittest.main()
