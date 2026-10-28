#!/usr/bin/env bash
# One edge-dit ed-cli generation on the Jetson Orin Nano (gap campaign E1/E2 functional gates).
# Usage (on the Jetson):  bash jetson_edgedit_smoke.sh <out-dir> [extra ed-cli args...]
# Fixed workload: distilled klein, 512x512, 4 steps, guidance 1, CFG 1, seed 0, the development prompt,
# the sd.cpp Jetson GGUF files, CUDA backend, flash attention, no cache, VAE tiling off.
# The extra args are the single labelled change for the rung (e.g. --auto-fit). Whole-system memory
# comes from tegrastats at 200 ms. jetson_memguard.sh kills the run early if memory runs out
# (memguard.json); the kernel OOM log is captured if the kernel kills it first.
set -uo pipefail
OUT=$1; shift
M=${MODELS:-$HOME/models/flux2-klein}
B=${EDGE_BIN:-$HOME/tools/edge-dit-20261010/build-cuda-performance/bin}
mkdir -p "$OUT"; cd "$OUT"
printf '%q ' "$@" > extra-args.txt
sha256sum "$M/flux-2-klein-4b-Q4_0.gguf" "${LLM:-$M/Qwen3-4B-Q4_K_M.gguf}" "$M/flux2-vae.safetensors" > model-sha256.txt
free -m > free-before.txt
tegrastats --interval 200 --logfile "$OUT/tegrastats.log" &
TP=$!
date -u +%FT%TZ > start.txt
START_LOCAL=$(date "+%F %T")
bash "$(dirname "$0")/jetson_memguard.sh" "$OUT" "$B/ed-cli" --backend cuda \
  --diffusion-model "$M/flux-2-klein-4b-Q4_0.gguf" --llm "${LLM:-$M/Qwen3-4B-Q4_K_M.gguf}" --vae "$M/flux2-vae.safetensors" \
  --type preserve --vae-tiling off --flash-attention --cache off \
  -W 512 -H 512 --steps 4 --guidance 1.0 --cfg-scale 1.0 --seed 0 --threads 4 \
  --prompt "A cat holding a sign that says hello world" --output "$OUT/output.png" \
  "$@" > ed-cli.log 2> ed-cli.stderr
echo $? > exit_code.txt
date -u +%FT%TZ > end.txt
kill $TP
journalctl -k --since "$START_LOCAL" 2>/dev/null | grep -iE "oom-kill|Out of memory" > kernel-oom.txt || true
grep -oE "RAM [0-9]+/[0-9]+MB" tegrastats.log | sort -t' ' -k2 -n | tail -1 > peak-ram.txt
grep -oE "SWAP [0-9]+/[0-9]+MB" tegrastats.log | sort -t' ' -k2 -n | tail -1 >> peak-ram.txt
cat memguard.json 2>/dev/null
echo "SMOKE_DONE exit=$(cat exit_code.txt)"
