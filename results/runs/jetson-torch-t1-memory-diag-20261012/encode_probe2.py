import gc, json, sys, threading, time, torch, transformers
def avail(): return int([l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1]) // 1024
snap = "/home/mfaruqi/hf/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots/e7b7dc27f91deacad38e78976d1f2b499d76a294"
log = open("results/t1-diag/encode_probe2.jsonl", "w")
def rec(**k):
    k.update(avail_mb=avail(), cuda_alloc_mb=round(torch.cuda.memory_allocated()/2**20), cuda_reserved_mb=round(torch.cuda.memory_reserved()/2**20), cuda_peak_mb=round(torch.cuda.max_memory_allocated()/2**20))
    log.write(json.dumps(k) + "\n"); log.flush()
stop = False
def watch():
    while not stop: rec(ev="sample"); time.sleep(0.1)
qc = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
te = transformers.Qwen3ForCausalLM.from_pretrained(snap, subfolder="text_encoder", quantization_config=qc, torch_dtype=torch.bfloat16, device_map={"": "cuda"})
tok = transformers.AutoTokenizer.from_pretrained(snap, subfolder="tokenizer")
gc.collect(); torch.cuda.synchronize(); rec(ev="loaded", te_dtype=str(te.dtype), attn=str(te.config._attn_implementation))
threading.Thread(target=watch, daemon=True).start()
text = tok.apply_chat_template([{"role": "user", "content": "A cat holding a sign that says hello world"}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
inp = tok(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
ids, mask = inp["input_ids"].cuda(), inp["attention_mask"].cuda()
rec(ev="tokens", real_tokens=int(mask.sum()))
with torch.no_grad():
    o = te(input_ids=ids, output_hidden_states=True, use_cache=False); torch.cuda.synchronize(); rec(ev="no_mask_end"); del o; gc.collect()
    o = te(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False); torch.cuda.synchronize(); rec(ev="with_mask_end")
stop = True
