# Contributing

DualComp accepts changes to the paper-bounded reference components,
documentation, tests, and release tooling. This repository is not a place for
private experiment artifacts or an undocumented reconstruction of the paper.

## Evidence rules

- Describe manuscript numbers as **paper-reported, transcribed, and not
  independently reproduced**.
- Cite the manuscript page, equation, figure, or table when changing a method
  contract or result transcription.
- Keep paper-omitted parameters explicit. Do not add a convenient default and
  call it the paper setting.
- Preserve conflicting table/prose values as separate transcribed evidence;
  do not silently choose or average them.

## Privacy and rights

Do not submit model weights, checkpoints, benchmark samples, annotations,
generated labels, raw predictions, prompts containing benchmark data, private
server paths, credentials, copied environments, or unlicensed third-party
source. Use synthetic tensors in tests.

## Development checks

Use Python 3.10 or 3.11 and run:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

Changes to the public API require focused contract tests. Packaging or release
changes must also pass the full-history hygiene check and the clean build checks
defined in `.github/workflows/ci.yml`.
