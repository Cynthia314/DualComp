# Paper method to code

This map links concepts described in the manuscript to the public source tree.
It does not assert that the repository can reproduce the reported numbers. The
paper omits several numerical settings, and no router weight is published.

## Core mapping

| Paper component | Public source | Responsibility | Evidence status |
|---|---|---|---|
| Parasitic Router attachment | `src/dualcomp/router.py` | Passes host text embeddings and masks to a caller-injected instruction encoder | Pooling/adapter architecture is absent from the paper and is not guessed |
| Shared Router MLP | `src/dualcomp/router.py` | Validates a caller-injected shared MLP representation | Layer count, widths, and activations are absent from the paper |
| Lambda/rho heads | `src/dualcomp/router.py` | Validates final values from two independent caller-injected heads | Head structure, Sigmoid placement, rho-range mapping, weights, and numeric `rho_min` are not published |
| SCSA local clustering | `src/dualcomp/scsa.py` | Groups tokens using caller-supplied predecessor neighborhoods and `tau` | The appendix-only neighborhood construction is not guessed |
| ViT attention equation | `src/dualcomp/attention.py` | Computes Eq. 2 from caller-projected CLS queries and visual keys | Host layout and head reduction remain caller responsibilities |
| SCSA cluster scoring | `src/dualcomp/scsa.py` | Uses cumulative CLS-to-patch attention | Paper specifies SUM scoring |
| SCSA representatives | `src/dualcomp/scsa.py` | Keeps peak tokens for small clusters and weighted representatives for large clusters | `theta_size(lambda)` is missing from the paper |
| IGSR local-difference anchors | `src/dualcomp/igsr.py` | Computes Eq. 4 from features and a caller-supplied local average, then selects unique maxima from caller-ordered regions | Region construction/order, tie handling, and AvgPool boundary behavior are absent from the paper |
| Text-aware structural modulation | `src/dualcomp/igsr.py` | Applies Eq. 5 to caller-supplied normalized geometric/text score maps | Normalization method and numeric `beta` are missing |
| Greedy topology completion | `src/dualcomp/igsr.py` | Connects anchors using caller-supplied neighbor offsets under decreasing Chebyshev distance | `N_n` construction is absent from the paper |
| Weighted dual-stream fusion | `src/dualcomp/fusion.py` | Concatenates `(1-lambda) * T_sem` and `lambda * T_geo` | Paper formula transcribed |
| Compression budget | `src/dualcomp/compressor.py` | Computes the continuous equations for one sample and validates a caller-supplied discrete policy | The library supplies no integer-rounding, remainder, or minimum-stream default and does not compose the streams |

## Router supervision and training

The paper gives the symbolic target
`lambda_gt = alpha * lambda_llm + (1-alpha) * lambda_rule`, but omits numeric
`alpha`, the label artifact, optimizer, schedule, split, seed, and checkpoint
identity. This release therefore publishes paper-bounded Router wiring only. It
does not publish label-generation, dataset, or training utilities, and no
private script default may be cited as a paper-reported hyperparameter.

## Host and evaluator integration

| Target | Public path | Release treatment |
|---|---|---|
| GeoLLaVA-8K | `integrations/geollava/README.md` | Status documentation only; no upstream source, patch, or weight is distributed |
| lmms-eval | `integrations/lmms_eval/README.md` | Status only; no runnable patch is distributed because the required host visual-path hook is unavailable for public release |

Qwen2.5-VL-7B is represented only in the paper result transcription. This
release does not claim a supported or independently evaluated transfer adapter.

## Descriptive evidence files

- `configs/paper_transcribed.json` records the method and experiment information
  present in the manuscript. Unknown values remain `null`.
- `results/reported/paper_results.json` records the manuscript's figures and
  tables with status `paper_reported_transcribed_not_reproduced`.

Neither file is an executable or verified reproduction profile.

The core modules are clean-room reference implementations derived from the
attached manuscript and general PyTorch knowledge. They do not copy or vendor
host-model or comparison-method code. See `PROVENANCE.md` for the recorded
module hashes and review boundary.

## Known method gaps

### Adaptive clustering threshold

The manuscript says `tau(lambda)` is monotonic but does not provide a complete
formula or constants in the attached paper.

The predecessor-neighborhood construction `N(i)` is also deferred to an
appendix that is not present. The public API therefore requires it explicitly.

The manuscript also gives no SCSA rule for zero-norm cosine inputs, equal
cosine candidates, equal cluster scores, equal peak-attention values, or an
output budget larger than the number of formed clusters. The reference API
fails on these ambiguous cases and never pads or truncates a successful SCSA
result; those are defensive contracts, not paper settings.

### Size-aware representation

The manuscript describes a lambda-adaptive `theta_size(lambda)` but does not
provide its formula. A fixed implementation value must not be represented as the
paper setting.

### Router and TASM constants

Numeric `alpha`, `beta`, and `rho_min` are absent from the attached manuscript.
They are required for an exact run.

The Router's compact-instruction encoder, shared-MLP and head details, rho-range
mapping, IGSR's subregion construction/order, normalization function and
AvgPool boundary rule, and greedy `N_n` offsets are likewise not specified.
They are caller-injected rather than inferred.

The IGSR fixed-capacity result, complete-path requirement, possible repeated
coordinates, padding/sentinel scheme, and Python-loop execution are documented
reference-API conventions, not paper settings or the fully parallel performance
implementation mentioned by the manuscript.

### Budget discretization

The continuous allocation equations are given, but integer rounding, minimum
per-stream budgets, and any cross-stream reallocation are not fully specified.

### Evaluation protocol

The split, sample count, prompt, answer parser, software versions, hardware,
precision, timing boundaries, and FLOPs procedure are not fully specified.

## Intentionally excluded

The public method does not include model or router weights, benchmark data,
generated labels, raw outputs, private experiment profiles, copied virtual
environments, caches, or unlicensed host-model patches.
