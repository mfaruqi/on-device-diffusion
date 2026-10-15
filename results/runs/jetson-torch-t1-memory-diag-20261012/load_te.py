import gc, json, sys, threading, time, torch, transformers
def avail(): return int([l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1]) // 1024
snap = "/home/mfaruqi/hf/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots/e7b7dc27f91deacad38e78976d1f2b499d76a294"
low = [avail()]; stop = False
def watch():
    while not stop: low[0] = min(low[0], avail()); time.sleep(0.05)
threading.Thread(target=watch, daemon=True).start()
a0 = avail(); t = time.time()
qc = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
m = transformers.Qwen3ForCausalLM.from_pretrained(snap, subfolder="text_encoder", quantization_config=qc, torch_dtype=torch.bfloat16, device_map={"": "cuda"})
torch.cuda.synchronize(); gc.collect(); stop = True
print(json.dumps({"avail_before_mb": a0, "avail_min_during_load_mb": low[0], "avail_after_mb": avail(),
  "load_s": round(time.time() - t, 1), "cuda_alloc_mb": round(torch.cuda.memory_allocated() / 2**20), "cuda_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20),
  "footprint_mb": round(m.get_memory_footprint() / 2**20)}, indent=1))
