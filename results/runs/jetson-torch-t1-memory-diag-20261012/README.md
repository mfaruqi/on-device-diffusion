# Jetson T1 memory diagnostics, 2026-10-12

Scripts run on the Jetson under scripts/jetson_memguard.sh; MemAvailable from /proc/meminfo.

| Script | Finding |
|---|---|
| load_transformer.py | GGUF Q4_0 transformer stays quantized: 100 Q4_0 tensors 1974.4 MB + 69 BF16 tensors 372.0 MB; 2365 MB on CUDA |
| load_te.py | Qwen3 NF4 (bitsandbytes, quantized while loading): 2721 MB on CUDA, footprint 2475 MB, load 99.8 s |
| load_all.py | transformer then encoder then VAE on CUDA: 843 MB MemAvailable left, 5263 MB CUDA allocated |
| gen_stages.py | first generation killed by the guard after text_encode began |
| encode_probe.py (L=64, 512) | random-token forward: MemAvailable drop ~450 MB at L=64 and ~940 MB at L=512 including logits; ~370-400 MB is first-call library/kernel loading outside the torch allocator |
| encode_probe2.py | pipeline-equivalent call (chat template, padded to 512, attention mask): ~815 MB drop; mask adds ~70 MB |

The prompt has 21 real tokens and is padded to 512 (pipeline default max_sequence_length).
encode_only.py (pipeline built with transformer=None, vae=None) was killed by the guard; cause not isolated.
