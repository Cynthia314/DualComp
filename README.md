# DualComp

DualComp is research code accompanying the manuscript **“Semantic-Geometric
Dual Compression: Training-Free Visual Token Reduction for Ultra-High-Resolution
Remote Sensing Understanding.”** It routes a visual-token budget between a
semantic stream and a geometric stream before a host multimodal language model.

> **Evidence status:** every benchmark and efficiency number in this public
> repository is transcribed from the manuscript. None is an independent rerun
> or measurement produced by this repository. The transcription is marked
> `paper_reported_transcribed_not_reproduced`, and this release must not be described as
> reproducing the paper.

This repository publishes no router weights, host-model weights, benchmark
images, annotations, generated labels, or raw model outputs.

## Method overview

1. A lightweight router predicts a duality factor, `lambda`, and a retention
   factor, `rho`, from frozen host-language-model text features.
2. SCSA aggregates spatially contiguous semantic regions.
3. IGSR selects geometry-sensitive anchors and completes topological paths.
4. Lambda-weighted semantic and geometric tokens are concatenated before the
   host multimodal projector and language model.

The manuscript describes the router as approximately one million parameters.
The compression path is described as training-free once the router has been
trained offline and frozen.

## Public code scope

The package contains clean-room, paper-derived reference components for Router
wiring, Eq. 2 attention, SCSA, IGSR, budget validation, and weighted
dual-stream fusion. It contains no pretrained state and no host-specific
execution hook. Because the paper does not disclose the instruction pooling or
shared-MLP architecture, output-head structures, or the mapping used to keep
`rho` above `rho_min`, the Router requires the encoder, shared MLP, and both
final-value heads to be injected by the caller; the library does not invent
them.

The paper gives continuous budget equations but not integer rounding. The
single-example budget helper consequently requires an explicit caller-provided
discrete budget policy and supplies no default. SCSA requires explicit `tau`,
`size_threshold`, and predecessor neighborhoods. IGSR requires explicit local
averages, normalized score maps, `beta`, region partition, neighbor offsets,
and a hard budget. These are deliberate fail-closed interfaces for details that
the attached PDF leaves unresolved.

For SCSA, ambiguous cosine, cluster-ranking, or peak-attention ties fail closed;
zero-norm tokens cannot participate in cosine comparisons; and `budget` means
an exact output cluster count rather than a padding or truncation request. These
are safety contracts for unspecified cases, not additional paper claims.

The IGSR routine is a transparent semantic reference, not the fully parallel
performance implementation mentioned by the paper. Region-label order,
candidate-tie order, complete-path retention, repeated coordinates, padding,
and insufficient-budget failure are documented API conventions rather than
paper settings. The package does not claim an end-to-end host integration.

## Paper-reported results

The following values are copied from the manuscript, not generated here:

| Paper-reported item | Value |
|---|---:|
| XLRS-Bench 13-subtask macro average | 53.1% |
| Main compression ratio | 42.4x |
| Tokens per grid | 14.2 |
| Average visual-token volume | 6.4k |
| LLM compute | 99.8 TFLOPs |
| Inference speed | 3.87 s/image |
| Qwen2.5-VL-7B transfer | 47.9% at 10.24x |

The complete transcription is stored in
[`results/reported/paper_results.json`](results/reported/paper_results.json).
Its source and status fields explicitly state that it is not a verified
measurement.

The paper transcription also preserves unresolved internal inconsistencies:

- the displayed DualComp subtask values average to about 53.2%, rather than the
  table's 53.1%;
- 1.52 seconds for visual encoding/compression plus 2.25 seconds for generation
  equals 3.77 seconds, not 3.87 seconds;
- the table and prose use slightly different token-volume and TFLOP values;
- one Top-K ablation value differs between the table and surrounding prose.

These observations are transcription checks only. They are not replacement
measurements.

## Why this is not a reproduction package

The attached manuscript does not provide enough information for an independent
rerun. Missing items include numerical values or formulas for `alpha`, `beta`,
`rho_min`, `tau(lambda)`, and the cluster-size threshold; router optimizer and
schedule details; Router instruction-pooling/shared-MLP details; SCSA
neighborhood construction; Router head structures and `rho` mapping; IGSR
normalization, pooling-boundary, subregion/order, tie, path-overlap, padding,
budget-remainder, neighbor, and parallelization rules; the exact data split,
prompt, parser, scorer, and evaluation script; hardware and precision; timing
boundaries; and a profiler-backed FLOPs recipe.
The manuscript refers to an appendix that is not present in the attached
19-page PDF.

[`configs/paper_transcribed.json`](configs/paper_transcribed.json) records these
unknowns as `null`. It is a descriptive record, not an executable experiment
profile. The release intentionally refuses to fill missing paper parameters
with undocumented defaults.

No executable paper profile is published. The clean-room component APIs require
the caller to provide values that the manuscript leaves unresolved, so private
or guessed defaults cannot be mistaken for paper settings.

## Repository layout

```text
src/dualcomp/            Clean-room paper-derived reference components
integrations/            Host/evaluator integration status; no runnable patch
configs/                 Paper configuration transcription only
results/reported/        Paper result transcription only
tests/                   CPU unit and contract tests
checkpoints/             Documentation only; no weights
```

## Installation

Python 3.10 or 3.11 is recommended for the packaged core.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
pytest -q
```

GeoLLaVA, `lmms_eval`, FlashAttention, benchmark data, and all model weights are
external to the core package. Review [THIRD_PARTY.md](THIRD_PARTY.md) and
[DATA.md](DATA.md) before using any external asset.

## API smoke check

This example checks only the exact weighted-concatenation formula and package
wiring. It is not an experiment or a paper-result reproduction.

```python
import torch

from dualcomp import lambda_weighted_concatenation

semantic = torch.ones(1, 2, 4)
geometric = torch.full((1, 1, 4), 2.0)
output = lambda_weighted_concatenation(semantic, geometric, duality=0.25)
assert output.shape == (1, 3, 4)
```

## Weights

No official or unofficial DualComp router weight is published by this release,
and no checkpoint loader, converter, checksum, or download identifier is
included. See [checkpoints/README.md](checkpoints/README.md) and
[MODEL_CARD.md](MODEL_CARD.md).

## Host integrations

The manuscript's primary host is GeoLLaVA-8K. The public GeoLLaVA integration
directory contains status documentation only because the audited upstream
source and model did not expose a clear license. No GeoLLaVA source, patch, or
weight is distributed here.

No `lmms-eval` patch is distributed. A candidate adapter was removed because,
without the legally cleared GeoLLaVA visual-path hook, it could attach DualComp
objects without actually executing compression. No task adapter, parser,
aggregator, or end-to-end evaluator is published.

Qwen2.5-VL-7B appears only as a paper-reported transfer result. This release
does not publish a supported Qwen adapter or a reproduced transfer artifact.

## Documentation

- [REPRODUCIBILITY.md](REPRODUCIBILITY.md): transcription scope and missing evidence
- [METHOD_TO_CODE.md](METHOD_TO_CODE.md): paper-method-to-code mapping
- [PROVENANCE.md](PROVENANCE.md): clean-room boundary and source hashes
- [MODEL_CARD.md](MODEL_CARD.md): unweighted router interface card
- [DATA.md](DATA.md): data exclusions and benchmark restrictions
- [THIRD_PARTY.md](THIRD_PARTY.md): dependency and licensing boundaries
- [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md): public-release gates
- [CONTRIBUTING.md](CONTRIBUTING.md): evidence, privacy, and test requirements

## Citation

Citation metadata is provided in [CITATION.cff](CITATION.cff). It intentionally
omits unverified publication identifiers.

## License

Original DualComp core code is licensed under Apache-2.0. That license does not
cover weights, datasets, annotations, images, generated labels, third-party
source, or third-party patches. See [LICENSE](LICENSE), [NOTICE](NOTICE), and
[THIRD_PARTY.md](THIRD_PARTY.md).
