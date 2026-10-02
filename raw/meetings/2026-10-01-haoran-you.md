# Meeting Memo - October 1, 2026

## Attendance

Mahad Faruqi, Prof. You

## Discussed

- [Profiling results](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion) for FLUX.2 [klein] 4B on A100 using PyTorch/diffusers and stable-diffusion.cpp, and Jetson Orin Nano using stable-diffusion.cpp only. Supporting records: [A100 engine comparison](https://github.com/mfaruqi/on-device-diffusion/blob/codex/readable-measurement-runners/experiments/compare-a100-pytorch-vs-sdcpp-flux-klein.md) and [Jetson baseline and profiles](https://github.com/mfaruqi/on-device-diffusion/blob/codex/readable-measurement-runners/experiments/jetson-flux-klein-003.md).
- The need to look more closely at the actual computation within each stage: kernels, memory movement, synchronization, and gaps between GPU activity.
- Expanding the baseline study to additional inference engines, starting with **edge-dit.cpp**, and testing optimizations already exposed by existing engines.
- Understanding what stable-diffusion.cpp's **auto-fit** selects before evaluating its effect on latency and memory. **EasyCache, DBCache, and related caching methods** are also candidates for controlled experiments.
- Showed [project github page](https://github.com/mfaruqi/on-device-diffusion/tree/codex/readable-measurement-runners) as well as [Wiki Agent](https://github.com/mfaruqi/on-device-diffusion/blob/codex/readable-measurement-runners/wiki/index.md) for keeping track of project state as a whole and showed how AI tools use its to ground answers as well as dynamically update the contents.

## To-Do

1. Profile the computation in greater detail. Break down text encoding, denoising, and VAE decoding into major operators and kernels. Investigate attention, matrix multiplication, convolutions, casts/copies, weight loading, and GPU idle intervals. Distinguish the computations from transfers and host-side overhead. This is especially relevant for Jetson.
2. Expand the current engine baselines. Evaluate [edge-dit.cpp](https://github.com/THU-MIG/edge-dit.cpp) first. An additional suggested engines was [TensorRT's diffusion pipeline](https://github.com/NVIDIA/TensorRT/tree/main/demo/Diffusion).
3. Audit and test the auto-fit and memory options. Inspect the pinned stable-diffusion.cpp implementation and logs to identify the placement, loading, offloading, or other decisions auto-fit actually makes. Record its resolved choices and compare against explicit fixed settings. Test supported options such as VAE tiling and CPU offload separately before combining them.
4. Evaluate existing caching methods. Start with EasyCache and DBCache. Also screen TaylorSeer, cache-dit, and Spectrum from the [sd.cpp caching options](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/caching.md). TeaCache and DiCache are additional research candidates where compatible implementations exist. Verify support in the pinned engine and target model, vary one policy at a time, and measure latency, cache memory, and output-quality effects against no caching.
5. **[Create or refine stacked stage-time bar charts.](https://wandb.ai/mfaruqi-purdue-university/on-device-diffusion/runs/00321d37da374213)** Organize bars by **engine + GPU/device + model**, with workload resolution and precision clearly labelled. Show text encoding, denoising, VAE decoding, and separately available postprocessing/other time.
6. Create roofline-style plots. showing the different runtimes of engine/compute/model.
