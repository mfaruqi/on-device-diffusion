import gc, json, sys, threading, time, torch, transformers
def avail(): return int([l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1]) // 1024
snap = "/home/mfaruqi/hf/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots/e7b7dc27f91deacad38e78976d1f2b499d76a294"
log = open("results/t1-diag/encode_probe.jsonl", "w")
def rec(**k):
    k.update(avail_mb=avail(), cuda_alloc_mb=round(torch.cuda.memory_allocated()/2**20), cuda_reserved_mb=round(torch.cuda.memory_reserved()/2**20), cuda_peak_mb=round(torch.cuda.max_memory_allocated()/2**20), t=round(time.time(),2))
    log.write(json.dumps(k) + "\n"); log.flush()
stop = False
def watch():
    while not stop: rec(ev="sample"); time.sleep(0.25)
qc = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
te = transformers.Qwen3ForCausalLM.from_pretrained(snap, subfolder="text_encoder", quantization_config=qc, torch_dtype=torch.bfloat16, device_map={"": "cuda"})
gc.collect(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); rec(ev="loaded")
threading.Thread(target=watch, daemon=True).start()
L = int(sys.argv[1])
ids = torch.randint(0, 150000, (1, L), device="cuda")
with torch.no_grad():
    rec(ev="fwd_begin", L=L)
    out = te.model(input_ids=ids, output_hidden_states=True)   # base model only: no lm_head logits
    torch.cuda.synchronize(); rec(ev="base_fwd_end", L=L)
    out2 = te(input_ids=ids, output_hidden_states=True)        # full CausalLM, as the pipeline calls it
    torch.cuda.synchronize(); rec(ev="causallm_fwd_end", L=L, logits_shape=list(out2.logits.shape), logits_dtype=str(out2.logits.dtype))
stop = True
