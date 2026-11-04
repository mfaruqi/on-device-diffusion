import gc, json, threading, time, torch, transformers, diffusers
def avail(): return int([l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1]) // 1024
snap = "/home/mfaruqi/hf/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots/e7b7dc27f91deacad38e78976d1f2b499d76a294"
low = [avail()]; stop = False
def watch():
    while not stop: low[0] = min(low[0], avail()); time.sleep(0.02)
threading.Thread(target=watch, daemon=True).start()
qc = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
te = transformers.Qwen3ForCausalLM.from_pretrained(snap, subfolder="text_encoder", quantization_config=qc, torch_dtype=torch.bfloat16, device_map={"": "cuda"})
pipe = diffusers.Flux2KleinPipeline.from_pretrained(snap, transformer=None, vae=None, text_encoder=te, torch_dtype=torch.bfloat16, local_files_only=True)
gc.collect(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
r = []
for i in range(2):
    low[0] = avail(); a = avail(); t = time.time()
    e = pipe.encode_prompt(prompt="A cat holding a sign that says hello world", device="cuda", max_sequence_length=512)
    torch.cuda.synchronize()
    r.append({"call": i, "avail_before_mb": a, "avail_min_during_mb": low[0], "avail_drop_mb": a - low[0], "s": round(time.time() - t, 2),
              "cuda_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20), "cuda_alloc_mb": round(torch.cuda.memory_allocated() / 2**20)})
    torch.cuda.reset_peak_memory_stats()
stop = True
print(json.dumps(r, indent=1))
