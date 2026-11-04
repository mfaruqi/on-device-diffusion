# Vendored research code

`shiftaddqwen/` contains the BCQ and GPTQ/BCQ fitting modules from
[Purdue-EcoAI/ShiftAddQwen](https://github.com/Purdue-EcoAI/ShiftAddQwen), pinned to
commit `12d995a51a514b3757e29213c5545004e2e0e347`.

See [provenance](shiftaddqwen/PROVENANCE.json) and the retained
[MIT license](shiftaddqwen/LICENSE). The adapter verifies source hashes before
fitting. Do not format or edit the vendored implementation as part of local style
cleanup; changes require explicit provenance and renewed numerical validation.

The FLUX adapter is in `src/shiftadd_flux/upstream_probe.py`. Vendoring these
modules does not imply that the complete ShiftAddQwen model or GPU execution
system has been reproduced for FLUX.
