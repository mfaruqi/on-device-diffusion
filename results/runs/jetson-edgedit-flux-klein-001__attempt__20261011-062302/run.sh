#!/usr/bin/env bash
cd /home/mfaruqi/results/jetson-edgedit-e1-20261011
sha256sum /home/mfaruqi/models/flux2-klein/flux-2-klein-4b-Q4_0.gguf /home/mfaruqi/models/flux2-klein/Qwen3-4B-Q4_K_M.gguf /home/mfaruqi/models/flux2-klein/flux2-vae.safetensors > model-sha256.txt
tegrastats --interval 200 --logfile /home/mfaruqi/results/jetson-edgedit-e1-20261011/tegrastats.log &
TP=$!
date -u +%FT%TZ > start.txt
/home/mfaruqi/tools/edge-dit-20261010/build-cuda-performance/bin/ed-cli --backend cuda --diffusion-model /home/mfaruqi/models/flux2-klein/flux-2-klein-4b-Q4_0.gguf --llm /home/mfaruqi/models/flux2-klein/Qwen3-4B-Q4_K_M.gguf --vae /home/mfaruqi/models/flux2-klein/flux2-vae.safetensors --type preserve --vae-tiling off --flash-attention --cache off -W 512 -H 512 --steps 4 --guidance 1.0 --cfg-scale 1.0 --seed 0 --threads 4 --prompt "A cat holding a sign that says hello world" --output /home/mfaruqi/results/jetson-edgedit-e1-20261011/output.png > ed-cli.log 2> ed-cli.stderr
echo $? > exit_code.txt
date -u +%FT%TZ > end.txt
kill $TP
