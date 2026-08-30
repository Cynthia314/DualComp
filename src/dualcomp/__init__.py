"""Paper-derived reference components for DualComp."""

from .attention import cls_to_patch_attention
from .compressor import (
    BudgetPolicy,
    ContinuousTokenBudget,
    TokenBudget,
    allocate_token_budget,
    continuous_token_budget,
)
from .fusion import lambda_weighted_concatenation
from .igsr import (
    IGSRResult,
    ParameterFreeIGSR,
    local_difference_saliency,
    recover_structure,
    select_region_anchors,
    structural_modulation,
    trace_greedy_path,
)
from .router import DualityAwareRouter
from .scsa import (
    SCSAResult,
    SpatiallyContiguousSemanticAggregator,
    spatially_contiguous_semantic_aggregate,
)

__all__ = [
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
]

__version__ = "0.1.0"
