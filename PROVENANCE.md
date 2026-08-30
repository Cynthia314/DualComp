# Source provenance

This document records how the paper-bounded public source tree was prepared. It is
an engineering provenance record, not legal advice and not evidence that the
paper results were independently reproduced.

## Release boundary

The release contains original Python reference components, documentation, and
two JSON transcriptions of the supplied manuscript. It contains no weights,
training or label-generation code, benchmark data, raw predictions, host-model
source, host patch, evaluator adapter, or private experiment configuration.

The implementation deliberately leaves paper-omitted values under caller
control. In particular, there are no defaults for the Router instruction
encoder/shared MLP, output heads, rho-range mapping, or `rho_min`; SCSA's
`tau(lambda)`, size threshold, predecessor neighborhoods, tie handling, or
exact-budget behavior; IGSR's local average, score normalization, `beta`,
subregion order, or neighbor offsets; and no integer token-budget rounding rule.
Reference API conventions that the paper does not specify are labeled as such
in source and documentation.

## Clean-room implementation record

Four isolated implementation tasks were used. Each implementer was instructed
to consult only the supplied manuscript and general PyTorch knowledge, and not
to inspect the pre-existing DualComp candidate, GeoLLaVA/RegProxy, VisionTrim,
or another implementer's source. The reviewed files were then installed
byte-for-byte; test imports were the only permitted test-file adjustment.

| Public file | Manuscript basis | SHA-256 |
|---|---|---|
| `src/dualcomp/router.py` | Paper-specified Router wiring with undisclosed encoder/MLP/heads injected, PDF pages 6-7 | `461737f1ee7492de0eb6035b0b9222c1cfbc569357fe9e9994eec3a71928e5cc` |
| `src/dualcomp/scsa.py` | SCSA and Eqs. 1-3 with undisclosed neighborhoods injected, PDF pages 7-8 | `87d6c410e2d8ac05a5c05683d28c30e08327820d8905d27cdbed02c8d6b9d38f` |
| `src/dualcomp/igsr.py` | IGSR and Eqs. 4-6 with undisclosed policies injected and API conventions labeled, PDF pages 8-10 | `4b4a37266de03ea3226fc0f5bc263182c8167633403e6291fa057d6217558ad9` |
| `src/dualcomp/attention.py` | CLS-to-patch attention, Eq. 2 | `3ad1d97dc23c296cdd61fcfc22c98b3c51e0431c8200891004eeb22877b5d5ba` |
| `src/dualcomp/fusion.py` | Weighted concatenation, PDF pages 9-10 | `8a22cb080ac3f26413788763af0f899f85e02efe14fd7255a5b20b3199255881` |
| `src/dualcomp/compressor.py` | Single-sample continuous budget equations and caller-supplied discrete-policy validation | `a77a6a8e57dc80abafe164bf8c9b53e9bf28921503a2d245147b5e7e4a68fdf6` |

`src/dualcomp/__init__.py` is original package-export wiring written during the
release review. No third-party implementation is copied or vendored in the
package.

## Paper evidence record

- `configs/paper_transcribed.json` contains only method and experiment details
  stated in the manuscript; unresolved values are `null`.
- `results/reported/paper_results.json` transcribes the paper's tables and
  figure values and is marked
  `paper_reported_transcribed_not_reproduced`.
- Document inconsistencies are retained as audit notes rather than silently
  corrected. They are not new experimental results.

## Third-party boundary

GeoLLaVA, lmms-eval, VisionTrim, VisionZip, XLRS-Bench, Qwen, CLIP, and their
weights or datasets are outside the distribution. `THIRD_PARTY.md` records the
audited references and terms. No source or data from those projects is present
in the public package.

## History policy

The public branch may retain only the repository's two reviewed placeholder
commits, which contained an Apache-2.0 license and a no-weights release notice.
The reviewed source tree is added as a new commit on that clean public history.
Development history that once contained rejected adapters, private result
artifacts, or candidate implementations must not be published. The final
release audit must inspect both the public tree and every object in its complete
history before upload.
