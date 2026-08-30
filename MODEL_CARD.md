# DualComp Router interface card

## Summary

The DualComp Router consumes frozen host-language-model text features and
predicts two controls:

- `lambda` in `[0, 1]`, which allocates tokens between semantic and geometric
  streams;
- `rho` in `[rho_min, 1]`, which controls the total retained budget.

The manuscript describes the router as approximately one million parameters,
trained offline and frozen during downstream inference.

The public constructor requires a caller-supplied instruction encoder, shared
MLP, lambda head, rho head, their intermediate dimensions, and `rho_min`. The
paper does not publish the pooling/adapter design, MLP layers, head structures,
rho-range mapping, or a checkpoint, so the library supplies none of them. Both
heads must return final values in the paper-stated ranges. The parameter count
of a public instance depends on all four injected modules and must not be
presented as the paper's trained Router.

## Weight availability

**No DualComp router weight is published by this repository.** There is no
official checkpoint, download link, checksum, or weight license in this
release. The `checkpoints/` directory contains documentation only.

No checkpoint-loading or conversion code is provided. A future weight release
must be separately licensed, reviewed, checksummed, and validated.

## Paper-described supervision

The manuscript defines the router target symbolically as:

```text
lambda_gt = alpha * lambda_llm + (1 - alpha) * lambda_rule
```

The attached manuscript does not state numeric `alpha`, the complete label
artifact, optimizer, learning rate, schedule, epochs, batch size, data split,
random seed, loss weights, or checkpoint identity. This release therefore
contains no label-generation or training scripts and supplies no undocumented
training defaults.

## Intended use

The router interface is intended for research on task-adaptive visual-token
compression with a compatible multimodal host. Appropriate uses include:

- inspecting and testing the routing architecture;
- controlled research using a user-owned, properly licensed checkpoint;
- studying semantic/geometric token-allocation mechanisms;
- implementing independently documented ablations.

## Out-of-scope use

The interface is not validated for:

- claims that this repository reproduces the manuscript's results;
- safety-critical navigation, disaster response, surveillance, or targeting;
- decisions about people or protected groups;
- commercial use of datasets restricted to non-commercial research;
- arbitrary host feature spaces without retraining and validation.

## Evaluation status

The public result file contains values transcribed from the manuscript. The
paper reports a 53.1% XLRS-Bench macro average for the main method and 47.9% at
10.24x for its Qwen2.5-VL-7B transfer row. These are not measurements produced
by this repository, and no published weight is available for an independent
rerun.

The transcription records internal inconsistencies in the paper tables and
prose. See `REPRODUCIBILITY.md`; do not silently reinterpret or correct the
reported values in downstream model cards.

## Limitations and biases

- Rule-based supervision inherits the assumptions and coverage gaps of the
  expert-designed linguistic rules.
- LLM-generated targets inherit the prompting behavior and biases of the model
  used for label generation.
- `lambda` and `rho` are control outputs, not calibrated probabilities.
- Compatibility depends on the host feature space, tokenizer, vision tower,
  multimodal projector, and preprocessing policy.
- Remote-sensing data can reveal sensitive locations and is subject to
  source-specific terms.
- Missing paper hyperparameters prevent exact independent evaluation.

## License

The repository's Apache-2.0 license covers original core code only. It grants no
rights to any external or user-provided weight, host model, dataset, annotation,
or generated label.
