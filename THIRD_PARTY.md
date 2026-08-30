# Third-party components and boundaries

This document records the known third-party boundary for the paper-only source
release. It is a provenance record, not legal advice. Licenses and model/data
terms must be rechecked immediately before publication.

## Inclusion policy

- The wheel contains only the `dualcomp` Python package under `src/dualcomp`.
- No host-model source, host-model weight, router weight, benchmark data, or
  generated router label is vendored.
- Paper comparisons and transfer rows do not make the compared projects part of
  the DualComp distribution.
- A patch against an upstream project remains subject to that project's rights
  and is not automatically covered by the DualComp Apache-2.0 license.

## Components

| Component | Paper/release role | License/status | Public treatment |
|---|---|---|---|
| [GeoLLaVA-8K](https://github.com/MiliLab/GeoLLaVA-8K) | Primary host named by the manuscript | No explicit repository license was found at audit time; a package classifier is not a license grant. | Documentation only. No source, patch, or weight is distributed. |
| [GeoLLaVA-8K model](https://huggingface.co/initiacms/GeoLLaVA-8K) | Host-model weights required for the paper path | No explicit model license was found in the reviewed model card. | Not distributed. Users must obtain any model directly under owner-supplied terms. |
| [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) | Candidate evaluation harness | Version 0.3.0 was reviewed; its license applies MIT to the main pipeline and Apache-2.0 to `lmms_eval/tasks` and `lmms_eval/models` | Not vendored or patched. No local task adapter or evaluator is distributed. |
| [VisionTrim](https://github.com/hanxunyu/VisionTrim/tree/308e73a9b8437882c5c1056a32dbc60fd41efd53) | Related TGVC-style text-guidance work | Apache-2.0 at audited commit `308e73a` | No source is vendored or modified in the clean-room public core; retain method-level acknowledgement. |
| [VisionZip](https://github.com/JIA-Lab-research/VisionZip/tree/8f86b55c6f000eb033e6912538af2dd7dcb30502) | Comparison method in the manuscript | Apache-2.0 at audited commit `8f86b55` | No source is vendored or derived; paper comparison only. |
| [XLRS-Bench](https://github.com/AI9Stars/XLRS-Bench/tree/828ac0ae8f200f6b05ac9ab12554caee6078e336) | Main benchmark and source domain for router supervision | The repository `LICENSE` is CC BY-NC-SA 4.0, while the current README describes annotations as CC BY-NC 4.0; underlying imagery has additional source-specific terms. | Terms must be resolved with the owner before redistribution. No images, annotations, questions, answers, prompts, choices, coordinates, or generated labels are distributed. See `DATA.md`. |
| Qwen2.5-VL-7B | Paper-reported transfer host | Code and model terms depend on the exact chosen revision. | Result transcription only. No adapter, source, or weight is distributed. |
| CLIP | Vision/text components referenced by the method | Subject to the selected implementation and weight terms | Not distributed. A future rerun must record exact model ID, revision, and license. |
| PyTorch | Core tensor runtime | See the installed distribution and `pyproject.toml` range | Dependency only; not vendored. |

## Integration status

The public `integrations/geollava/` directory contains status documentation
only. No GeoLLaVA source or derivative patch may be published until the upstream
rights holder supplies clear terms or permission.

The `integrations/lmms_eval/` directory contains status documentation only. A
candidate wrapper-only adapter was removed because it did not implement the
host visual-path hook required to execute DualComp. No `lmms-eval` source or
derivative patch is distributed.

## Adding another component

Any new dependency or integration must record:

- canonical project and model URLs;
- exact version, tag, commit, or model revision;
- code license and separate weight/data terms;
- whether code is linked, patched, copied, or only compared;
- required attribution or NOTICE text;
- whether the artifact is included in the wheel, repository, or neither.
