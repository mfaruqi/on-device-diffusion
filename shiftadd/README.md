# ShiftAdd experiments for FLUX.2 Klein

Research subproject for compressing the **distilled FLUX.2 Klein 4B transformer**.
It includes packed weight conversion, calibration-aware fitting, reconstruction
loaders, activation quantization references, and paired image diagnostics.
The text encoder and VAE are outside the weight-conversion scope.

**Status:** weights are saved and image diagnostics work. Image runners reconstruct
dense BF16 weights; they do **not** demonstrate packed GPU memory, latency or energy
savings. There is no custom GPU kernel or production inference backend here.
The parent repository's [instructions](../AGENTS.md) and experiment protocols apply.

## Start here

| Path | Purpose |
|---|---|
| [Code guide](src/shiftadd_flux/README.md) | Conversion, loaders, arithmetic and evaluation entry points |
| [Configuration guide](configs/README.md) | Configuration families, path setup and safe reproduction |
| [Documentation index](docs/README.md) | Methods, execution plans and measurement limits |
| [Tests](tests/README.md) | Numerical and representation checks |
| [Upstream source](vendor/README.md) | Pinned ShiftAddQwen source, provenance and license |
| `artifacts/`, `runs/` | Local-only weights, images and working outputs; excluded from Git |
| [Experiment registry](../wiki/experiments.md) | Durable records and small raw evidence in the parent repository |

## Methods and what was learned

- **Greedy sign planes:** initial weight-only baseline; full-transformer images fail.
- **Joint radix:** restricted power-of-two sign-plane fit; recognizable calibration
  images. Lossless exponent compaction reduces metadata storage.
- **Mixed precision:** 21 matrices use five planes and the others four. Selection
  uses weight error, not a learned activation-sensitivity policy.
- **Conventional INT4:** same-source symmetric quantization control. It is competitive
  with radix; the experiments do not establish a radix quality advantage.
- **Upstream GPTQ/BCQ:** pinned ShiftAddQwen fitter applied to two slices and two
  complete projections. Its u8/shared-FP16 scales differ from the restricted radix
  family. This is **not** a full-transformer ShiftAddQwen reproduction.
- **Activations:** grouped A4/A8 arithmetic is numerically checked. All-linear
  fake-quantized image tests execute floating GEMMs; they are quality diagnostics.

[Current evidence](docs/current-campaign.md) links completed experiments and limits.
No formal held-out quality evaluation has been performed.

## Saved transformer storage

All 109 matrix weights plus unchanged transformer tensors, excluding reports,
text encoder and VAE. These are logical deployment file sizes, not live memory
or the total disk occupied by multiple saved candidates.

| Bundle | GiB | Reduction from BF16 |
|---|---:|---:|
| Source BF16 | 7.219 | — |
| Conventional INT4 | 1.918 | 73.4% |
| Radix p4 | 1.918 | 73.4% |
| Compact radix p4 | 1.833 | 74.6% |
| Compact mixed p4/p5 | 1.900 | 73.7% |

[Storage evidence and exact accounting](../experiments/compare-shiftadd-transformer-storage.md).
Weights are not included in this repository or uploaded publicly. Some local
bundles share hard links: treat all completed artifacts as immutable.

## Install and run CPU checks

Run from the repository root. Use a Python 3.10+ environment compatible with your
PyTorch installation; CUDA installations must match the target device/runtime.

```bash
python3 -m venv .venv-shiftadd
source .venv-shiftadd/bin/activate
python -m pip install -e './shiftadd[dev]'
python -m unittest discover -s shiftadd/tests -v
ruff check shiftadd/src shiftadd/tests
ruff format --check shiftadd/src shiftadd/tests
```

Image runners additionally need the `images` extra and a compatible Diffusers
version with `Flux2KleinPipeline`. Plotting uses `plots`; W&B export uses `publish`
and the parent repository's exporter. The CPU tests require no checkpoint or GPU.
Recorded runs retain their actual environment versions; optional extras are not
an exact reproduction lockfile.

## Reproduce a conversion

1. Obtain the checkpoint through your authorized model access. The recorded
   source revision is `e7b7dc27f91deacad38e78976d1f2b499d76a294`.
2. Copy a [configuration](configs/README.md), set `source` to the transformer
   safetensors file, and choose a **new** output directory. Recorded configs contain
   author-local paths and must be adapted for another machine.
3. Run the corresponding module. For example:

```bash
python -m shiftadd_flux.convert --config /path/to/new-radix-config.json
```

Conversion refuses an existing output directory. Keep previous evidence, including
failed runs. Calibration-based fitting additionally requires captured activations
and the pinned vendor source; see the configuration guide.

For images, use `pipeline_smoke` with a prepared local checkpoint and existing
artifacts. Run CUDA jobs on an allocated GPU, never an HPC login node. Supplied
Slurm scripts are Gilbreth-specific examples, not portable launch defaults.

## Interpretation rules

Storage reduction does not imply faster inference. BF16 latent distance does not
replace prompt correctness or a formal image-quality score. Calibration prompts
remain separate from held-out evaluation. Numerical arithmetic references are not
performance kernels. See [performance estimates](docs/performance-estimates.md)
and the [energy measurement contract](docs/energy-contract.md).
