# Checkpoints

This paper-only release publishes **no DualComp router weight** and no host-model
weight. This directory is documentation-only.

Do not commit:

- router or host-model state dicts;
- optimizer or scheduler checkpoints;
- tensor embeddings or caches;
- private filenames, checksums, or download links that imply an official model
  release.

The package includes no checkpoint loader, converter, checksum, model ID, or
download link. A user-provided checkpoint is not an official DualComp release
and cannot establish reproduction of the paper results.

Any future weight publication must be a separate, explicitly approved release
with its own model card, license, data and host-model provenance, tensor-only
format, checksum, and independent validation. No such release is part of this
repository version.
