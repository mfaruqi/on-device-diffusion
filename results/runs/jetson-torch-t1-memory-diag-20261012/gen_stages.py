exec(open("results/t1-diag/load_all.py").read())
import functools
def wrap(obj, attr, name):
    f = getattr(obj, attr)
    @functools.wraps(f)
    def w(*a, **k):
        mark(name + ":begin"); r = f(*a, **k); torch.cuda.synchronize(); mark(name + ":end"); return r
    setattr(obj, attr, w)
wrap(pipe, "encode_prompt", "text_encode")
wrap(pipe.vae, "decode", "vae_decode")
pipe.transformer.register_forward_hook(lambda m, a, o: mark("transformer_step:end"))
pipe.set_progress_bar_config(disable=True)
img = pipe(prompt="A cat holding a sign that says hello world", height=512, width=512, num_inference_steps=4, guidance_scale=1.0,
           max_sequence_length=512, generator=torch.Generator("cuda").manual_seed(0)).images[0]
mark("done"); img.save("results/t1-diag/gen.png")
