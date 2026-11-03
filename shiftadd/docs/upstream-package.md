# Complete upstream weight packages

`package_upstream` combines an existing full-row upstream fit with every
unconverted tensor from its pinned BF16 transformer. It does not refit weights
or quantize the remainder. Packages stay under ignored `shiftadd/artifacts/`;
do not commit or upload model weights publicly.

## Saved Klein coverage

The [verified package](../../experiments/m5-flux-klein-shiftadd-020.md)
contains every transformer tensor, with these representations:

| Weight group | Matrices | Representation |
|---|---:|---|
| Dual-stream image Q/K/V, five blocks | 15 | Four-plane upstream BCQ |
| Dual-stream text Q/K/V, five blocks | 15 | Four-plane upstream BCQ |
| Single-stream fused QKV/MLP, twenty blocks | 20 | Four-plane upstream BCQ |
| Attention output projections | 30 | Original BF16 |
| Dual-stream image/text feed-forward matrices | 20 | Original BF16 |
| Embedding, modulation, timestep and final projections | 9 | Original BF16 |

The other 60 non-matrix tensors are unchanged BF16. Thus all Q/K/V weights are
included in the upstream conversion, but not every transformer matrix is
converted. The twenty fused matrices also contain MLP rows. This coverage
comes from the complete tensor inventory in the package record.

## Assemble fitted subsets

Before packaging, the two source fit bundles can be combined reproducibly:

```bash
PYTHONPATH=shiftadd/src python -m shiftadd_flux.assemble_upstream \
  --config shiftadd/configs/upstream-assembly-all-attention-001.json
```

Each selection explicitly lists modules and their source artifact directory.
The tool requires completed full-row fits and independent reload verification,
checks shared calibration/checkpoint settings and every copied file hash, and
rejects duplicate modules. Plane counts may differ by module; assembly does
not change fitted values. Its output is a fresh loader bundle, not a new fit.
The example writes a separate `upstream-bcq-all-attention-rebuilt-001` directory;
point a copy of the packaging config at it to package that reproduction.

## Create

From the repository root, with the subproject installed or its source on
`PYTHONPATH`:

```bash
PYTHONPATH=shiftadd/src python -m shiftadd_flux.package_upstream \
  --config shiftadd/configs/upstream-transformer-package-001.json
```

The configuration preserves original machine paths. Copy it and adapt the
paths before reproducing elsewhere. The output must not exist. The tool checks
the source checkpoint hash, packed-file hashes and complete row coverage. It
writes the BF16 remainder and verifies every saved remainder tensor against
the source before marking the package complete. Errors leave a failed status
and all partial files available for inspection.

## Contents

| File | Purpose |
|---|---|
| `*.npz` | Existing packed upstream matrices, copied without changes |
| `fallback.safetensors` | All other transformer tensors, unchanged BF16 |
| `transformer-config.json` | Architecture configuration, when supplied |
| `manifest.json` | Complete tensor inventory, representations, file sizes and hashes |
| `config.json`, `summary.json` | Original fit settings and evidence required by the decoder |
| `package-config.json`, `status.json` | Packaging provenance and completion status |

Source paths in the provenance files do not need to exist when loading.
The package contains transformer weights only. Text encoder, tokenizer,
scheduler, VAE and pipeline configuration remain separate dependencies.
This is a research package, not a Hugging Face `from_pretrained` directory.

## Load

Construct the correct architecture with real BF16 parameter storage, then load
the package. For the saved Klein package:

```python
import json
from pathlib import Path

import torch
from diffusers import Flux2Transformer2DModel
from shiftadd_flux.package_upstream import load_package

bundle = Path("shiftadd/artifacts/upstream-bcq-transformer-package-001")
architecture = json.loads((bundle / "transformer-config.json").read_text())
with torch.device("meta"):
    transformer = Flux2Transformer2DModel.from_config(architecture).to(torch.bfloat16)
transformer.to_empty(device="cpu")
audit = load_package(transformer, bundle, chunk_rows=128)
```

The loader checks parameter names, shapes, dtypes and file hashes. It reads
the BF16 remainder and reconstructs the packed matrices in row chunks. It does
not open the original checkpoint. The model ends up with dense BF16 weights;
this path does not provide packed GPU execution or reduced resident weight
memory. The architecture must be allocated before loading; meta parameters
are rejected by `load_package`.

## Accounting

The report separates weight-file bytes from total package bytes, including
metadata. Both are logical file sizes, not filesystem allocation or runtime
memory. The source checkpoint and other experiment bundles are retained, so
total workspace disk usage exceeds the deployment package size. The packaging
operation itself temporarily needs memory for the unconverted tensor payload.

## Optional packed-array cache

`load_package(..., cache_packed_arrays=True)` caches the packed arrays once per
matrix while keeping dense reconstruction chunked. The default is `False`,
preserving earlier experiment behavior. A [CPU loading diagnostic](../../experiments/m5-flux-klein-shiftadd-load-002.md)
measures this option on one fused matrix and verifies full-package numerical
agreement. This is not an inference-speed result.
