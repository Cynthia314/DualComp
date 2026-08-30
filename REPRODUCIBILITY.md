# Reproducibility and evidence status

## Scope

This is a paper-transcription release, not an independently reproduced result
release. The public evidence artifacts are:

- `configs/paper_transcribed.json`: method and experimental settings copied
  from the attached manuscript, with missing values retained as `null`;
- `results/reported/paper_results.json`: tables and figure values copied from
  the manuscript, marked `paper_reported_transcribed_not_reproduced`.

No benchmark result, latency, token count, or FLOPs value in the public tree was
measured by this repository. No router or host-model weights are published.

## Paper-reported headline values

| Item | Value reported by the paper | Independent rerun in this repository |
|---|---:|---|
| XLRS-Bench macro average | 53.1% | No |
| Compression ratio | 42.4x | No |
| Tokens per grid | 14.2 | No |
| Average visual-token volume | 6.4k | No |
| LLM compute | 99.8 TFLOPs | No |
| Inference speed | 3.87 s/image | No |
| Qwen2.5-VL-7B transfer | 47.9% at 10.24x | No |

The paper describes the XLRS score as an unweighted macro average over 13
subtasks. The repository does not contain the predictions or immutable raw
counts needed to recompute the reported table independently.

## Internal checks on the paper transcription

The transcription retains the manuscript values exactly and records the
following inconsistencies rather than silently correcting them:

1. The displayed DualComp subtask percentages average to approximately 53.2%,
   not the table's 53.1%.
2. The paper lists 1.52 seconds for visual encoding/compression and 2.25 seconds
   for LLM generation; their sum is 3.77 seconds, not the reported 3.87-second
   end-to-end value.
3. The efficiency table uses 13.8k visual tokens and 99.8 TFLOPs, while nearby
   prose uses 14.0k and 99.75 TFLOPs.
4. The Top-K ablation row and the page-13 prose disagree on at least one score.

These are document consistency checks. The repository does not substitute new
measurements or choose which manuscript value is scientifically correct.

## Missing information that blocks an independent rerun

The paper transcription records the following unresolved items:

### Router

- numeric `alpha` in the hybrid supervision target;
- `rho_min`;
- the compact-instruction encoder/pooling design, shared-MLP layer details,
  head structures, Sigmoid placement, and rho-range mapping;
- optimizer, learning rate, weight decay, schedule, epochs, batch size, split,
  random seed, and loss weights;
- label-generation artifact and checkpoint identity.

### SCSA and IGSR

- concrete `tau(lambda)` formula and constants;
- concrete `theta_size(lambda)` formula;
- SCSA predecessor-neighborhood construction `N(i)`;
- numeric TASM `beta`;
- the score-normalization method, AvgPool boundary rule, region partition and
  anchor order, tie handling, and greedy neighborhood `N_n`;
- path overlap/deduplication, unused budget, padding, and parallelization rules;
- exact integer rounding and minimum per-stream token rules.

### Dataset and evaluator

- sample count and evaluated split;
- exact prompt template, answer parser, and scoring implementation;
- exact evaluation script and model revisions;
- legally redistributable data manifest.

### Efficiency protocol

- hardware;
- precision and attention implementation;
- software versions;
- warm-up and repetition counts;
- timing boundaries;
- maximum generation length;
- profiler or reviewed FLOPs formula.

The manuscript refers to implementation details in an appendix, but the
attached 19-page PDF ends with references and contains no appendix.

## Descriptive configuration only

`configs/paper_transcribed.json` is intentionally non-executable. Missing paper
parameters remain `null`; they must not be replaced with private or guessed
defaults and then labeled as the paper configuration.

No executable paper profile is included. The clean-room component APIs require
all paper-missing values explicitly and therefore do not silently substitute
private or guessed defaults.

Any utility that consumes this record should fail clearly when a required value
is absent. A successful synthetic smoke test proves only that code executes; it
does not validate a paper result.

## Requirements for a future independent rerun

An independent reproduction claim requires all of the following:

1. obtain the missing paper parameters and exact router checkpoint under
   distributable terms;
2. freeze legally cleared GeoLLaVA, vision-tower, tokenizer, evaluator, and data
   revisions;
3. publish the exact prompt, parser, crop policy, generation settings, seeds,
   and resolved configuration;
4. run the full benchmark from a clean checkout without reused outputs;
5. retain per-subtask correct/total counts and, where dataset terms permit,
   per-sample identifiers and predictions;
6. measure latency with explicit timing boundaries and warm-up/repetition rules;
7. report FLOPs from a reviewed formula or profiler and label the method;
8. have an independent maintainer verify the environment, checksums, and
   aggregation;
9. publish the rerun separately from the immutable paper transcription;
10. update the status only after evidence exists.

Until then, the correct wording is **“paper-reported”**, **“transcribed”**, and
**“not independently reproduced.”**
