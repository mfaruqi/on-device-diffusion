import gc, json, threading, time, torch, transformers, diffusers
def avail(): return int([l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1]) // 1024
snap = "/home/mfaruqi/hf/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots/e7b7dc27f91deacad38e78976d1f2b499d76a294"
low = [avail()]; stop = False
def watch():
    while not stop: low[0] = min(low[0], avail()); time.sleep(0.05)
threading.Thread(target=watch, daemon=True).start()
def mark(s):
    gc.collect(); print(json.dumps({"step": s, "avail_mb": avail(), "min_so_far_mb": low[0], "cuda_alloc_mb": round(torch.cuda.memory_allocated()/2**20)}), flush=True)
mark("start")
tr = diffusers.Flux2Transformer2DModel.from_single_file("/home/mfaruqi/models/flux2-klein/flux-2-klein-4b-Q4_0.gguf",
     quantization_config=diffusers.GGUFQuantizationConfig(compute_dtype=torch.bfloat16), config=snap, subfolder="transformer", torch_dtype=torch.bfloat16)
mark("transformer_on_cpu")
tr.to("cuda"); torch.cuda.synchronize(); mark("transformer_on_cuda")
qc = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
te = transformers.Qwen3ForCausalLM.from_pretrained(snap, subfolder="text_encoder", quantization_config=qc, torch_dtype=torch.bfloat16, device_map={"": "cuda"})
mark("text_encoder_nf4_on_cuda")
pipe = diffusers.Flux2KleinPipeline.from_pretrained(snap, transformer=tr, text_encoder=te, torch_dtype=torch.bfloat16, local_files_only=True)
mark("pipeline_built")
pipe.vae.to("cuda"); torch.cuda.synchronize(); mark("vae_on_cuda")
