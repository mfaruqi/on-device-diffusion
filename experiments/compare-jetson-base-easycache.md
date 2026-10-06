---
type: experiment-record
id: compare-jetson-base-easycache
status: complete
device: jetson-orin-nano
engine: stable-diffusion-cpp
runs: [jetson-flux-klein-base-001__repeat__20261005-042014, jetson-flux-klein-base-002__baseline__20261005-024556, jetson-flux-klein-base-001__attempt__20261005-021728, jetson-flux-klein-base-002__attempt__20261005-023028, jetson-flux-klein-base-001__baseline__20261005-005357]
updated: 2026-10-05
---

# Compare: Jetson Base no-cache and EasyCache

## Question

Does approximate step reuse reduce work while preserving useful output on the longer-denoising workload? RQ1, Week1–2 image baseline evidence. A completed same-binary full no-cache repeat removes the earlier harness-build confound. Sequential-capture conditions remain uncontrolled, so the observed ratio is descriptive rather than an isolated causal estimate.

## Runs

| Role | Run and saved config | Record |
|---|---|---|
| No-cache control, new harness | [jetson-flux-klein-base-001__attempt__20261005-021728](../results/runs/jetson-flux-klein-base-001__attempt__20261005-021728/config.json) | [reference](jetson-flux-klein-base-001.md) |
| EasyCache attempt, same harness | [jetson-flux-klein-base-002__attempt__20261005-023028](../results/runs/jetson-flux-klein-base-002__attempt__20261005-023028/config.json) | [cache](jetson-flux-klein-base-002.md) |
| No-cache full reference and pixel source | [jetson-flux-klein-base-001__baseline__20261005-005357](../results/runs/jetson-flux-klein-base-001__baseline__20261005-005357/config.json) | [reference](jetson-flux-klein-base-001.md) |

| EasyCache full repeated run | [jetson-flux-klein-base-002__baseline__20261005-024556](../results/runs/jetson-flux-klein-base-002__baseline__20261005-024556/config.json) | [cache](jetson-flux-klein-base-002.md) |

## Matching

| Setting | Attempt pair |
|---|---|
| Device/engine | Same Jetson25W, sd.cpp19bbbca, binary SHA2562d66fa04053c02b441d61a84d24412537a16f9099d6cb31b80dcca3c8cc1f896 |
| Weights | Identical component pins: Base Q4_0, Qwen3 Q4_K_M, original VAE |
| Workload |512×512,50 Euler steps, flux2 schedule, guidance4, seed0, batch1, identical prompt |
| Protocol | First plus one observation, no warm-ups; one-second whole-system memory samples |
| Residency | Eager disk parameters, segmentation/prefetch on, no automatic fitting or conditioning cache |
| Intended variable | No step reuse versus EasyCache threshold0.2/start0.15/end0.95 |

Confounds: sequential captures with uncontrolled filesystem-cache, background-process and swap state. Same seed and engine fix the noise-generation procedure; no external latent tensor was supplied. The earlier repeated reference uses an older harness and1+3+10; it supplies the paired reference pixels, whose equality with the newer control was [verified](../results/runs/jetson-flux-klein-base-001__attempt__20261005-021728/control-validation.json). Its timing is not mixed into the two-observation comparison.

## Attempt results

Values come from the [control summary](../results/runs/jetson-flux-klein-base-001__attempt__20261005-021728/summary.json) and [cache summary](../results/runs/jetson-flux-klein-base-002__attempt__20261005-023028/summary.json). Second observations are not warm baseline medians.

| Metric | No-cache second, ms | EasyCache second, ms | Cache minus control, ms |
|---|---:|---:|---:|
| wall_ms | 321034.480 | 184900.394 | -136134.086 |
| text_encode_ms | 23153.013 | 22399.402 | -753.611 |
| denoise_ms | 293037.660 | 157859.932 | -135177.728 |
| vae_decode_ms | 4553.690 | 4384.475 | -169.215 |

Whole-capture system RAM peaks:control6.041GiB, cache5.988GiB; these include load and are not isolated cache-allocation measurements. Swap state differs; do not infer a cache memory saving.

## Image and execution diagnostics

[Paired-pixel receipt](../results/runs/jetson-flux-klein-base-002__attempt__20261005-023028/cache-validation.json): RGB images differ; PSNR31.204dB, computed over decoded8-bit RGB values with peak255. Both observations within each attempt are pixel-identical. A subsequent [local CPU diagnostic](../results/runs/jetson-flux-klein-base-002__attempt__20261005-023028/quality-diagnostics.json) measured calibrated LPIPS AlexNet v0.1=0.008252, with same-image distance0. Inputs were full-resolution RGB, scaled to[-1,1], no crop/resize; evaluator package versions and both weight hashes are saved. The initial capture receipt's unavailable field predates this separate evaluation. Visually the cached output retains a coherent cat and legible sign.

No-cache logs contain100 transformer executions per image; EasyCache contains50 and reports25/50 skipped steps. This confirms reuse at the configured50-step guidance-four workload. The engine's printed2x estimate is not a measured end-to-end speedup.

## Earlier repeated runs, different harness builds

The [no-cache summary](../results/runs/jetson-flux-klein-base-001__baseline__20261005-005357/summary.json) and [cache summary](../results/runs/jetson-flux-klein-base-002__baseline__20261005-024556/summary.json) both use1+3+10. Workload, weights and engine revision match as above; the no-cache reference predates the cache-capable harness. The new harness's two-generation no-cache control preserved pixels, but cannot establish identical repeated latency. Filesystem-cache/swap/background conditions were not randomized or controlled.

| Metric | No-cache median [min–max], ms | Cache median [min–max], ms | Cache minus reference, ms |
|---|---:|---:|---:|
| wall_ms | 325631.833 [313668.333–330741.966] | 184192.778 [181321.881–190636.573] | -141439.055 |
| text_encode_ms | 24357.311 [15685.698–25337.666] | 22132.515 [13932.880–25001.993] | -2224.796 |
| denoise_ms | 298587.185 [289659.113–301166.262] | 158403.877 [154372.333–164174.193] | -140183.308 |
| vae_decode_ms | 4539.275 [3598.732–4728.997] | 4454.548 [3859.684–4591.680] | -84.727 |

The observed median ratio is1.768× (reference/cache), a descriptive comparison. The cache maximum is below the reference minimum, so the numerical range rule is met. This earlier pair alone does not support selection because its binaries differ; the completed same-binary comparison below addresses that mismatch. No second independently qualified option currently exists for this Base workload.

Whole-capture RAM peaks:reference6.027GiB, cache5.996GiB. Both use one-second system samples including load; the small difference does not establish cache memory savings. All fourteen cache images match the evaluated attempt pixels ([validation receipt](../results/runs/jetson-flux-klein-base-002__baseline__20261005-024556/cache-validation.json)), so the existing paired PSNR/LPIPS measurements remain applicable to that same image pair.

## Same-binary repeated comparison

[No-cache repeat](../results/runs/jetson-flux-klein-base-001__repeat__20261005-042014/config.json), [summary](../results/runs/jetson-flux-klein-base-001__repeat__20261005-042014/summary.json), [matching receipt](../results/runs/jetson-flux-klein-base-001__repeat__20261005-042014/cache-comparison.json). This repeat and the full EasyCache run use the same binary SHA2562d66fa04053c02b441d61a84d24412537a16f9099d6cb31b80dcca3c8cc1f896. Component pins, workload, protocol1+3+10, placement, segmentation, prefetch and all non-cache arguments match exactly. Only the step-cache policy differs.

| Metric | No-cache median [min–max], ms | Cache median [min–max], ms | Cache minus reference, ms |
|---|---:|---:|---:|
| wall_ms | 328719.445 [313962.972–331504.639] | 184192.778 [181321.881–190636.573] | -144526.668 |
| text_encode_ms | 24192.235 [16156.323–24600.976] | 22132.515 [13932.880–25001.993] | -2059.720 |
| denoise_ms | 299599.784 [289221.908–301971.807] | 158403.877 [154372.333–164174.193] | -141195.907 |
| vae_decode_ms | 4598.830 [3588.229–4890.237] | 4454.548 [3859.684–4591.680] | -144.282 |

Observed median ratio:1.785×. EasyCache's measured maximum remains below the no-cache minimum. Both run images match their corresponding images in the existing PSNR/LPIPS evaluation, so the diagnostic pair is unchanged. Under the campaign rule for approximate caches (record diagnostic metrics, do not impose the exact-option PSNR threshold), EasyCache now qualifies as a candidate for later combinations. This is not formal quality approval; no second independently qualified Base option exists yet.

Whole-capture one-second RAM peaks:no-cache6.015GiB, cache5.996GiB. This small difference does not isolate cache-allocation cost. Sequential filesystem-cache, swap and background-process conditions remain confounds; the comparison is qualitative about attribution despite the large observed separation. The reduction from100to50transformer executions per generation is direct evidence of work skipped.

## Limits and next evidence

One prompt/seed, no held-out evaluation and uncontrolled sequential capture conditions. The new full comparison matches harness builds, while earlier builds remain explicitly separated. No formal quality eligibility or combined-policy result. Test lazy loading independently before considering a combined cache/residency configuration.

[W&B paired-image analysis](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/jetson-base-easycache-quality-20261005).
