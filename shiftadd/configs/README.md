# Configuration guide

Configs are labelled research inputs, not portable zero-configuration examples.
They preserve original paths and settings. Copy before adapting paths, checkpoint
locations or output directories; never overwrite a completed run.

| Filename family | Entry point / use |
|---|---|
| `transformer-*.json` | `convert`, `convert_uniform` or `compact_radix`, according to the declared format |
| `activation-*.json`, grouped diagnostics | Captured-input fitting and arithmetic probes; inspect module `--help` |
| `upstream-kv-activation-error-*.json` | `upstream_activation_probe --config ...`; fixed weights, grouped A4/A8 numerical errors |
| `upstream-load-*.json` | `upstream_load_probe --config ...`; fresh-process CPU loading time and RSS |
| `upstream-transformer-package-*.json` | `package_upstream --config ...`; complete saved weights with a BF16 remainder |
| `upstream-bcq-*.json` | `upstream_probe --config ...`, followed by `verify_upstream --config ...` |
| `a100-*.json` | Paired pipeline, calibration or activation-capture runners; the companion Slurm script names the module |
| `*.slurm` | Site-specific offline GPU jobs for Purdue Gilbreth |

Common conversion fields: `source` is the transformer safetensors file;
`source_revision` pins its model revision; `output` must be a fresh directory;
`group_size` and `planes` define the representation. Not every module accepts
every field: use the matching runner and preserve the original experiment settings.

Upstream fits also require `capture` (a directory containing `activations.npz`),
`vendor`, selected `modules`, `fit_calls`, `scale_format` and damping settings.
`row_policy: all` fits complete projections; sampled slices are not deployable
full-layer replacements. Capture data are local-only and must be regenerated.

Image cases name their artifact directory, format and exact tensor selection.
A `null` selection usually means all tensors in that bundle. The first case must
be unmodified BF16. Activation settings declare modules, bit width, group size
and, where present, expected calls per module. Do not silently reduce coverage.

To reproduce on another device, record a separately labelled configuration and
its actual environment. Gilbreth examples pin an A100 node for paired comparisons;
account, partition, paths and Python interpreter need site-specific setup.
