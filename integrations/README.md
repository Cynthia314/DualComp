# Host integrations

DualComp is inserted after the CLIP vision tower and before the host multimodal
projector. The standalone package and CPU tests do not prove host-level result
reproduction.

- `lmms_eval/` records why no runnable evaluator patch is distributed yet.
- `geollava/` records the required base commit and the publication blocker.

Apply integrations only to clean, pinned upstream checkouts. Never overwrite a
virtual environment's `site-packages` directory.
