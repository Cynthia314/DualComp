# DualComp

Research code for **“Semantic-Geometric Dual Compression: Training-Free Visual
Token Reduction for Ultra-High-Resolution Remote Sensing Understanding.”**

> Results in this repository are transcribed from the manuscript and are not
> independently reproduced.

## Overview

DualComp routes a visual-token budget between semantic and geometric streams:

- **Router** predicts task-adaptive `lambda` and `rho` controls.
- **SCSA** aggregates spatially contiguous semantic regions.
- **IGSR** retains geometry-sensitive anchors and completes structural paths.
- **Fusion** concatenates the two streams using lambda-aware scaling.

The repository provides a paper-bounded reference implementation. Parameters
omitted by the manuscript remain explicit, and no end-to-end host-model adapter
is claimed.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Python 3.10 and 3.11 are supported.

## Minimal example

```python
import torch
from dualcomp import lambda_weighted_concatenation

semantic = torch.ones(1, 2, 4)
geometric = torch.full((1, 1, 4), 2.0)
output = lambda_weighted_concatenation(semantic, geometric, duality=0.25)

assert output.shape == (1, 3, 4)
```

## Paper-reported results

| Item | Reported value |
|---|---:|
| XLRS-Bench average | 53.1% |
| Compression ratio | 42.4x |
| Inference speed | 3.87 s/image |
| Qwen2.5-VL-7B transfer | 47.9% at 10.24x |

These values are copied from the manuscript and are not independent reruns.

## Citation and license

Citation metadata is in [CITATION.cff](CITATION.cff). Code in this repository is
released under [Apache-2.0](LICENSE); external assets remain subject to their
own terms.
