"""One-image feasibility capture; this does not report benchmark metrics."""
import datetime, hashlib, json, os, re, subprocess, traceback
from pathlib import Path
from PIL import Image
assert os.environ.get('SLURM_JOB_ID'), 'Inference requires an allocated Slurm job'
cfg=json.loads(Path('configs/a100-edgedit-flux-klein-bf16-sample-smoke.json').read_text())
out=Path('output')/('edge-sample-smoke-'+os.environ['SLURM_JOB_ID']); out.mkdir(parents=True,exist_ok=False)
def save(name,value): (out/name).write_text(json.dumps(value,indent=2)+'\n')
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
start=now();save('status.json',{'status':'running','started_utc':start})
save('config.json',cfg)
try:
    binary=Path(cfg['engine']['binary']); source=binary.parents[2]
    commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    assert commit==cfg['engine']['commit']
    assert Path(cfg['model']['path']).is_dir()
    env={'job':os.environ['SLURM_JOB_ID'],'hostname':os.uname().nodename,'engine_commit':commit,'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'gpu':subprocess.check_output(['nvidia-smi'],text=True),'submodules':subprocess.check_output(['git','-C',str(source),'submodule','status'],text=True)}
    save('environment.json',env)
    assert 'A100-PCIE-40GB' in env['gpu'], 'Unexpected GPU variant'
    prompt=out/'prompt.txt'; prompt.write_text(cfg['workload']['prompt']+'\n')
    cmd=[str(binary),'--model',cfg['model']['path'],*cfg['cli_arguments'],
         '--prompt_file',str(prompt),'--output_dir',str(out/'engine'),
         '--warmup',str(cfg['protocol']['warmup']),'--repeat',str(cfg['protocol']['repeat'])]
    mapping={'width':'width','height':'height','steps':'num_steps','guidance':'guidance_scale',
             'cfg_scale':'cfg_scale','seed':'seed','sampler':'sampler','scheduler':'scheduler'}
    for key,flag in mapping.items():cmd.extend(['--'+flag,str(cfg['workload'][key])])
    save('command.json',cmd)
    with (out/'engine.log').open('w') as log:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
    im=Image.open(out/'engine/imgs/img_000000.png').convert('RGB');assert im.size==(1024,1024)
    log=(out/'engine.log').read_text()
    passes=re.findall(r'\[ed-sample\] pass (\d+)/(\d+)\s+1/1\s+seed=0\s+([0-9.]+)s',log)
    assert len(passes)==2 and [(a,b) for a,b,_ in passes]==[('1','2'),('2','2')],passes
    assert all(float(seconds)>0 for _,_,seconds in passes)
    save('interface-check.json',{'passes':passes,'timing_source':'upstream steady_clock around ed_generate_image; stdout rounded to milliseconds','retained_images':'final repeat only; upstream overwrites index filename','stage_audit':'pending inspection','baseline_eligible':False})
    save('image-check.json',{'dimensions':list(im.size),'rgb_sha256':hashlib.sha256(im.tobytes()).hexdigest(),'visual_inspection':'pending','quality_eligible':False})
    save('status.json',{'status':'complete','started_utc':start,'finished_utc':now(),'scope':'Two repeats in one ed-sample context; final image only. Stage markers, effective settings and timing boundaries require inspection before benchmark integration.'})
except BaseException:
    save('status.json',{'status':'failed','started_utc':start,'finished_utc':now(),'error':traceback.format_exc()});raise
