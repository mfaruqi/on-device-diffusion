# Meeting Memo - October 7, 2026

## Attendance

Mahad Faruqi, Prof. You

## Discussed

- Reviewed the new FLUX.2 klein distilled and Base benchmarking results on A100 and Jetson Orin Nano, including EasyCache, exact prompt-conditioning reuse, and memory/loading options. Compared their effects on text encoding, denoising, and total generation time using the [stage-by-stage timing charts in W&B](https://forge.coreweave.com/wandb/mfaruqi-purdue-university/on-device-diffusion/runs/c6af35f4845e4833)
- Discussed how caching benefits depend on the workload and device. Distinguished approximate denoising reuse from caching text embeddings for repeated identical conditioning, and kept performance improvements separate from image-quality validation
- Discussed shift-add computation and sensitivity-aware quantization as possible diffusion compiler optimizations including different precision choices by layer or denoising step and static versus dynamic quantization

## To-Do

- Continue benchmarking engines and their optimizations comprehensively
- Create 3×2 figures comparing engines on our Jetson results using the [edge-dit presentation](https://github.com/THU-MIG/edge-dit.cpp/blob/main/docs/performance-4090.md) as a reference
- Then investigate shift-add and selective quantization for diffusion using the [ShiftAdd paper](https://arxiv.org/abs/2406.05981).
