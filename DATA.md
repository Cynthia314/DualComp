# Data policy

## No data or weights are distributed

This repository does not distribute benchmark images, annotations, questions,
answers, generated router labels, cached embeddings, raw predictions, router
weights, or host-model weights. Empty data, result, and checkpoint directories
are documentation or layout placeholders only.

Users must obtain every external asset directly from its owner, review the
current terms, and configure paths outside the Git checkout. Possession of this
code does not grant permission to download, use, modify, or redistribute any
dataset or model.

## XLRS-Bench

The manuscript reports its primary evaluation on XLRS-Bench. At audit time:

- The reviewed XLRS repository `LICENSE` file was CC BY-NC-SA 4.0, while its
  current README described annotations as CC BY-NC 4.0. This inconsistency must
  be resolved with the owner before any redistribution.
- Images came from multiple sources with different conditions.
- Some sources restricted use to research or non-commercial purposes.
- Some sources restricted redistribution even when annotations were available.

The XLRS repository is the starting point, but users must also follow each
underlying image provider's terms. Sources noted in the reviewed benchmark
materials include DOTA/Google Earth imagery, ITCVD, MiniFrance/HRSCD, Toronto,
and Potsdam. Their terms are not interchangeable.

Do not copy benchmark images into documentation, tests, examples, release
archives, containers, or issue reports. Use synthetic tensors for unit tests and
an independently cleared image for any visual demonstration.

## Router supervision labels

The paper defines the target symbolically as:

```text
lambda_gt = alpha * lambda_llm + (1 - alpha) * lambda_rule
```

The attached manuscript does not state numeric `alpha` or identify a complete
label artifact. Label-generation and router-training utilities are intentionally
excluded from this release because the audited internal versions contained
benchmark-style prompts and non-paper experiment records.

Generated JSON can contain benchmark questions, answer choices, ground truth,
image references, and derived labels. Those artifacts must not be committed.
Before any separate label release:

1. obtain explicit permission from benchmark and underlying data owners;
2. remove machine-local paths and personal metadata;
3. document the generating model, revision, prompt template, rule-engine
   version, seed, and generation date;
4. publish a dedicated data card and label license;
5. expose non-commercial and attribution restrictions at download time.

## Local configuration

Dataset locations must be supplied at runtime, for example through an
`XLRS_DATA_ROOT` environment variable or an untracked local configuration file.
The repository must not contain a machine-specific absolute path.

An illustrative local layout is:

```text
external-data/
└── xlrs-bench/
    ├── annotations/
    └── images/        # only when permitted by each image provider
```

`external-data/` is ignored by Git. The official benchmark instructions
determine the actual layout; this release publishes no dataset adapter.

## Paper result transcription

`results/reported/paper_results.json` contains values copied from the
manuscript. It contains no claim that this repository generated the values. Its
status is `paper_reported_transcribed_not_reproduced`.

No raw predictions or rerun outputs are public. A future independent result
must remain separate from the immutable paper transcription and must label its
metric definitions, data revision, configuration, and evidence source.

## Future model artifacts

Any future weight publication is outside this release and requires a separate
license, model card, provenance review, tensor-only format, checksum, and
independent validation. The current public repository intentionally contains no
weight identifier or download offer.
