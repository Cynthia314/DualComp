# Paper-only public-release checklist

This repository may be released only as source code plus a paper transcription.
It is **not** a reproduced-results or model-weight release.

## Public evidence scope

- [x] Mark every public result as
      `paper_reported_transcribed_not_reproduced`.
- [x] Describe all benchmark and efficiency numbers as paper-reported or
      transcribed.
- [x] Record the manuscript's internal arithmetic and prose/table
      inconsistencies without replacing its values.
- [x] Keep unknown paper settings as `null` rather than filling them with
      undocumented defaults.
- [x] Confirm the public `configs/` directory contains only paper transcription
      and clearly non-verified paper-method specification material, with no
      private or non-paper experiment profile.
- [x] Confirm the public `results/` directory contains only the immutable paper
      transcription and no independently observed or raw experiment result.
- [x] Confirm no README, badge, release note, or package description uses
      “reproduced,” “verified result,” or equivalent wording.

## Weights and model artifacts

- [x] Publish no router weight, host-model weight, embedding cache, optimizer
      checkpoint, or model binary.
- [x] Keep `checkpoints/` documentation-only.
- [x] Ignore `.pt`, `.pth`, `.ckpt`, `.safetensors`, `.bin`, and model/cache
      directories.
- [x] Scan the final Git tree and full history to confirm no weight was ever
      staged.
- [x] Confirm documentation contains no private checkpoint filename, checksum,
      or implied download offer.

## Missing paper information

- [x] Document missing numeric `alpha`, `beta`, `rho_min`, `tau(lambda)`, and
      `theta_size(lambda)`.
- [x] Document missing router optimizer, schedule, split, seed, label artifact,
      and checkpoint identity.
- [x] Document missing dataset split, prompt, parser, and exact evaluator.
- [x] Document missing hardware, precision, timing boundaries, repetitions, and
      profiler-backed FLOPs procedure.
- [x] Record that the attached paper refers to an appendix that is absent.
- [ ] Obtain author-approved missing information before creating any executable
      paper reproduction profile.

## Rights and licensing

- [x] Scope Apache-2.0 to original DualComp core code.
- [x] Exclude weights, data, annotations, generated labels, third-party source,
      and third-party patches from the core license.
- [ ] Obtain a clear GeoLLaVA source/model license or written permission before
      publishing any GeoLLaVA patch or weight.
- [ ] Complete a line-level provenance review against acknowledged research
      implementations.
- [x] Preserve required Apache/MIT copyright and NOTICE text.
- [ ] Recheck every code, model, and dataset license on the release date.

## Source and integration hygiene

- [x] Remove every machine-specific absolute path, credential, private endpoint,
      and personal metadata item from the public tree and Git history.
- [x] Ensure the public package exports only the intended paper-method path.
- [x] Ensure public configuration does not expose a private run profile.
- [x] Remove the incomplete wrapper-only `lmms-eval` patch and end-to-end run
      script; retain status documentation only.
- [x] Keep the GeoLLaVA directory documentation-only until permission exists.
- [x] Confirm scripts require explicit runtime paths and do not ship local
      defaults.
- [x] Confirm no virtual environment, cache, compiled extension, or copied model
      tree is included.

## Data

- [x] Publish no XLRS image, annotation, question, answer, generated label, or
      raw prediction.
- [x] Document that users obtain data directly under source-specific terms.
- [x] Review all documentation assets and issue templates for accidental data
      redistribution.
- [ ] Review any future aggregate artifact for embedded prompts, paths, or
      dataset-derived content.

## Tests and packaging

- [x] Router shape, mask, range, and parameter tests pass.
- [x] SCSA clustering, SUM scoring, representative, and budget tests pass.
- [x] IGSR anchor, text-modulation, topology, order, and budget tests pass.
- [x] Weighted-fusion, single-sample budget-policy, and public-API contract tests pass.
- [x] Paper transcription schema and consistency-audit tests pass.
- [ ] A clean CPU CI matrix passes on Python 3.10 and 3.11.
- [x] `python -m build` and `twine check dist/*` succeed from a clean clone.
- [x] Wheel contents contain only intended core package files.
- [x] CI large-file, secret-pattern, and absolute-path checks pass.

## Publication wording

- [ ] Release title says “source code and paper transcription,” not
      “reproduction package.”
- [ ] Release notes state that no weights are published.
- [ ] Release notes state that all numbers are paper-reported and not
      independently rerun.
- [x] `CITATION.cff` contains only verified public identifiers.
- [ ] Scientific owner, code owner, and data/license reviewer approve the final
      tree.

## Sign-off

| Role | Name | Date | Decision |
|---|---|---|---|
| Scientific owner |  |  |  |
| Code owner |  |  |  |
| Data/license reviewer |  |  |  |
