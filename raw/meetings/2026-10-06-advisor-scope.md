# Advisor project overview and scope confirmation - October 6, 2026

## Project overview (as provided by Prof. Haoran You)

On-Device Diffusion Models

Recent advancements have sparked a surge in applications powered by diffusion models, ranging from image and video generation and editing [1,2,3,4,5] to world model simulation [6] and text-based reasoning [7,8]. Despite their capabilities, these models demand significant computational resources, often necessitating cloud-based hosting on large-scale GPU clusters [9]. Despite progress made by initiatives like DreamLite [10] in device-specific deployment and SVDQuant [13,14] in quantization, there remains a critical need for a universal on-device suite. This suite must support diverse hardware—including NVIDIA, Intel, AMD, and Apple silicon—and operate seamlessly across various platforms, including macOS, Linux, Windows, Android, and web browsers [12]. Consequently, this project aims to develop a universal engine for diffusion models, utilizing open-weight models like Flux and Wan to enable efficient and ubiquitous cross-platform deployment.

[1] High-Resolution Image Synthesis with Latent Diffusion Models, CVPR'22
[2] https://github.com/compvis/stable-diffusion
[3] https://github.com/black-forest-labs/flux
[4] Wan: Open and Advanced Large-Scale Video Generative Models, arXiv'25
[5] https://github.com/hao-ai-lab/fastvideo
[6] https://github.com/next-state/open-dreamer
[7] Large Language Diffusion Models, NeurIPS'25
[8] ELF: Embedded Language Flows, arXiv'26
[9] HiLo-Token: Input-Adaptive High-Low Frequency Token Compression for Efficient Image Editing, arXiv'26
[10] DreamLite: A Lightweight On-Device Unified Model for Image Generation and Editing, arXiv'26
[11] https://github.com/ByteVisionLab/DreamLite
[12] https://github.com/mlc-ai/mlc-llm
[13] https://github.com/nunchaku-ai/nunchaku
[14] SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models, ICLR'25

## Scope question and answer

Question: "The overview describes an MLC-style universal diffusion engine; my proposal focuses on a planner on top. Should the thesis deliver (a) a TVM-compiled FLUX.2 klein pipeline running on CUDA, Metal and WebGPU, with (b) the reuse/precision/residency planner as the research contribution on top? And which non-NVIDIA/Apple target matters most to you: web, Android, or AMD/Intel?"

Answer (relayed by Mahad Faruqi, 2026-10-06): yes.

Schedule guidance (Mahad Faruqi, 2026-10-06): keep the schedule optimistic; work may roll over past the stated weeks if needed.
