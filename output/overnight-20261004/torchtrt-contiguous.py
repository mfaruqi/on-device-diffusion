"""Bounded compile gate with real inputs; not an inference benchmark runner."""
import datetime, hashlib, json, logging, os, time, traceback
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID'), 'Inference requires an allocated GPU job'
out=Path('output')/('torchtrt-contiguous-'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True,exist_ok=False)
def save(name,value): (out/name).write_text(json.dumps(value,indent=2,default=str)+'\n')
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
started=now();save('status.json',{'status':'running','started_utc':started})
cfg=json.loads(Path('configs/a100-flux-klein-torchtrt-contiguous-smoke.json').read_text());save('config.json',cfg)
phase='imports'
try:
    import torch, torch_tensorrt, tensorrt, diffusers, transformers
    from diffusers import Flux2KleinPipeline
    assert torch.cuda.is_available()
    assert torch.__version__=='2.5.1+cu124' and torch_tensorrt.__version__=='2.5.0'
    gpu=torch.cuda.get_device_name(0);assert 'A100-PCIE-40GB' in gpu
    save('environment.json',{'torch':torch.__version__,'cuda_build':torch.version.cuda,'tensorrt':tensorrt.__version__,'torch_tensorrt':torch_tensorrt.__version__,'diffusers':diffusers.__version__,'transformers':transformers.__version__,'gpu':gpu,'hostname':os.uname().nodename,'scope':'Isolated cu124 environment; original reference used cu121.'})
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('torch_tensorrt').setLevel(logging.DEBUG)
    phase='load';t=time.perf_counter()
    pipe=Flux2KleinPipeline.from_pretrained(cfg['model']['repo_id'],revision=cfg['model']['revision'],torch_dtype=torch.bfloat16,local_files_only=True).to('cuda')
    torch.cuda.synchronize();load={'load_total_s':time.perf_counter()-t};save('load.json',load)
    pipe.set_progress_bar_config(disable=True)
    captured=[];keys=['hidden_states','encoder_hidden_states','timestep','img_ids','txt_ids']
    def capture(module,args,kwargs):
        if captured:return
        assert not args and kwargs.get('guidance') is None
        assert not kwargs.get('joint_attention_kwargs') and kwargs.get('return_dict') is False
        captured.extend(kwargs[k].detach().clone() for k in keys)
    handle=pipe.transformer.register_forward_pre_hook(capture,with_kwargs=True)
    phase='eager_image';w=cfg['workload']
    with torch.no_grad():
        image=pipe(prompt=w['prompt'],height=w['height'],width=w['width'],num_inference_steps=w['num_inference_steps'],guidance_scale=w['guidance_scale'],max_sequence_length=w['max_sequence_length'],generator=torch.Generator('cuda').manual_seed(w['seed']),output_type='pil').images[0]
    handle.remove();assert image.size==(1024,1024);image.save(out/'eager.png')
    save('eager-image.json',{'rgb_sha256':hashlib.sha256(image.tobytes()).hexdigest(),'dimensions':list(image.size)})
    assert len(captured)==5
    save('inputs.json',{k:{'shape':list(x.shape),'dtype':str(x.dtype),'device':str(x.device)} for k,x in zip(keys,captured)})
    class Region(torch.nn.Module):
        def __init__(self,transformer):super().__init__();self.transformer=transformer
        def forward(self,hidden,encoder,timestep,img_ids,txt_ids):
            return self.transformer(hidden_states=hidden,encoder_hidden_states=encoder,timestep=timestep,img_ids=img_ids,txt_ids=txt_ids,guidance=None,return_dict=False)[0]
    region=Region(pipe.transformer).eval();original_args=tuple(captured)
    assert cfg['integration']['input_layout']=='contiguous'
    args=tuple(x.contiguous() for x in original_args)
    assert all(torch.equal(x,y) for x,y in zip(original_args,args))
    save('input-layout.json',{k:{'source_stride':list(x.stride()),'source_contiguous':x.is_contiguous(),'compile_stride':list(y.stride()),'compile_contiguous':y.is_contiguous(),'values_equal':True} for k,x,y in zip(keys,original_args,args)})
    assert {p.dtype for p in region.parameters()}=={torch.bfloat16}
    with torch.no_grad():
        expected=region(*original_args).detach()
        phase='export';torch.cuda.synchronize();t=time.perf_counter()
        ep=torch.export.export(region,args,strict=False)
        torch.cuda.synchronize();load['export_s']=time.perf_counter()-t;save('load.json',load)
        (out/'export-graph.txt').write_text(str(ep.graph))
        phase='compile';t=time.perf_counter();settings=dict(cfg['optimizations']['torch_tensorrt']);settings.pop('version');settings['enabled_precisions']={torch.bfloat16}
        compiled=torch_tensorrt.dynamo.compile(ep,arg_inputs=args,**settings)
        torch.cuda.synchronize();load['compile_s']=time.perf_counter()-t;save('load.json',load)
        modules=[{'name':n,'type':type(m).__module__+'.'+type(m).__name__} for n,m in compiled.named_modules()]
        engines=[m for m in modules if m['type'].startswith('torch_tensorrt.dynamo.runtime.')]
        save('coverage.json',{'modules':modules,'trt_runtime_modules':engines,'scope':'Structural partition inventory; compiler log records supported/unsupported ops. Not a runtime fraction or full-pipeline coverage.','mixed_execution_allowed':True})
        (out/'compiled-graph.txt').write_text(str(compiled.graph))
        assert engines,'No TensorRT runtime regions were produced; this is not successful acceleration'
        phase='compiled_forward';actual=compiled(*args);torch.cuda.synchronize()
        assert actual.shape==expected.shape and actual.dtype==expected.dtype and torch.isfinite(actual).all()
        delta=actual.float()-expected.float()
        save('numerical-diagnostics.json',{'max_abs_error':delta.abs().max().item(),'rmse':delta.square().mean().sqrt().item(),'output_dtype':str(actual.dtype),'scope':'Single transformer input; no whole-image quality eligibility or speedup claim.'})
    save('status.json',{'status':'complete','started_utc':started,'finished_utc':now(),'scope':'Representative transformer forward only; full-pipeline hooks, image quality and repeated timing remain unvalidated.'})
except BaseException:
    save('status.json',{'status':'failed','phase':phase,'started_utc':started,'finished_utc':now(),'error':traceback.format_exc()});raise
