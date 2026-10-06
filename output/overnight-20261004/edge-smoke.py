"""One-image feasibility capture; this does not report benchmark metrics."""
import datetime, hashlib, json, os, subprocess, traceback
from pathlib import Path
from PIL import Image
assert os.environ.get('SLURM_JOB_ID'), 'Inference requires an allocated Slurm job'
cfg=json.loads(Path('configs/a100-edgedit-flux-klein-bf16-smoke.json').read_text())
out=Path('output')/('edge-smoke-'+os.environ['SLURM_JOB_ID']); out.mkdir(parents=True,exist_ok=False)
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
    cmd=[str(binary),'--model',cfg['model']['path'],*cfg['cli_arguments']]
    for k,v in cfg['workload'].items():
        if k!='batch_size':cmd.extend(['--'+k.replace('_','-'),str(v)])
    cmd.extend(['--output',str(out/'image.png')]);save('command.json',cmd)
    with (out/'engine.log').open('w') as log:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
    im=Image.open(out/'image.png').convert('RGB');assert im.size==(1024,1024)
    save('image-check.json',{'dimensions':list(im.size),'rgb_sha256':hashlib.sha256(im.tobytes()).hexdigest(),'visual_inspection':'pending','quality_eligible':False})
    save('status.json',{'status':'complete','started_utc':start,'finished_utc':now(),'scope':'Single image CLI execution; logs, effective settings and image require inspection before benchmark integration.'})
except BaseException:
    save('status.json',{'status':'failed','started_utc':start,'finished_utc':now(),'error':traceback.format_exc()});raise
