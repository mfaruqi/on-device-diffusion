import torch
from torch.profiler import profile, record_function, ProfilerActivity
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
).to("cuda")

prompt = "An horse riding an astronaut on the moon"

# Warmup so profiler doesn't capture model load
_ = pipe(prompt, num_inference_steps=4, height=512, width=512)
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    with record_function("flux_inference"):
        image = pipe(prompt, num_inference_steps=4, height=512, width=512).images[0]
image.save("profile_output.png")

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
prof.export_chrome_trace("flux_trace.json")
print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
