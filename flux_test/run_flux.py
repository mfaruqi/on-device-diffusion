import torch
from diffusers import FluxPipeline
import time

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",  # or FLUX.1-dev
    torch_dtype=torch.bfloat16
)
pipe = pipe.to("cuda")

prompt = "A futuristic on-device AI chip glowing in a dark lab"

# Warmup
_ = pipe(prompt, num_inference_steps=4, height=512, width=512)

# Timed run
torch.cuda.synchronize()
start = time.time()
image = pipe(prompt, num_inference_steps=4, height=512, width=512).images[0]
torch.cuda.synchronize()
end = time.time()

image.save("output.png")
print(f"Inference time: {end - start:.3f}s")
print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
