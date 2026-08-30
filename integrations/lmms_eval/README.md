# lmms-eval integration status

No `lmms-eval` patch is distributed in this paper-only release.

An earlier candidate adapter could attach method objects to the LongVA wrapper,
but it did not implement the required GeoLLaVA visual-path hook.
Publishing it as a runnable integration would allow an uncompressed host-model
run to be mistaken for DualComp. It has therefore been removed.

A future integration requires all of the following before publication:

- clear redistribution terms for the exact GeoLLaVA host revision;
- a reviewed hook before feature selection/projector execution that, for each
  sample, explicitly allocates a discrete budget, invokes public SCSA and IGSR
  with every paper-missing input supplied, and then applies explicit fusion;
- explicit CLIP text guidance and ViT attention projections;
- a fail-closed one-sample integration test proving that compression actually
  changes the visual-token sequence;
- an exact evaluator, prompt, parser, data revision, and timing protocol.

No task adapter, raw-result aggregator, or end-to-end evaluator is included.
