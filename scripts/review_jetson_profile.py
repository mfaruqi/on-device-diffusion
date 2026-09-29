#!/usr/bin/env python3
"""Review imported single-generation Jetson NVTX artifacts without modifying originals."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import struct
import zlib
from analyze_profile import analyze, render_md


def write(p, value):
    p.write_text(json.dumps(value, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    r = p.parse_args().run_dir
    o = r / 'originals'
    read = lambda name: json.loads((o / name).read_text())
    rows = [json.loads(x) for x in (o / 'results.jsonl').read_text().splitlines()]
    assert len(rows) == 2 and all(x['ok'] for x in rows)
    load, g = rows
    assert g['progress_calls'] == 5 and len(g['t_step_end']) == 4 and g['cond_cache_hits'] == 0
    for name in ['config.json', 'environment.json', 'command.json', 'results.jsonl']:
        shutil.copyfile(o / name, r / name)
    write(r / 'artifact-manifest.json', {str(x.relative_to(o)): {'bytes': x.stat().st_size, 'sha256': hashlib.sha256(x.read_bytes()).hexdigest()} for x in sorted(o.rglob('*')) if x.is_file()})
    c = sqlite3.connect('file:' + str((o / 'trace.sqlite').resolve()) + '?mode=ro', uri=True)
    assert c.execute('pragma integrity_check').fetchone()[0] == 'ok'
    ranges = c.execute('SELECT start,end,coalesce(text,s.value) FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id WHERE end IS NOT NULL ORDER BY start').fetchall()
    expected = ['load', 'generate', 'text_encode'] + ['denoise_step_'+str(i) for i in range(4)] + ['vae_decode']
    assert [x[2] for x in ranges] == expected
    windows = {name:(a,b) for a,b,name in ranges}
    ga,gb = windows['generate']
    children = [x for x in ranges if x[2] not in ['load','generate']]
    assert all(ga <= a < b <= gb for a,b,n in children)
    assert all(x[1] <= y[0] for x,y in zip(children,children[1:]))
    activities=[]
    for table in ['CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET']:
        activities.extend((table,a,b) for a,b in c.execute('select start,end from '+table))
    crossing=[(kind,a,b,n) for kind,a,b in activities for x,y,n in children if x<=a<y<b]
    assert not crossing, 'GPU events cross stage ends; inspect before attributing'
    prof=r/'profile';prof.mkdir(exist_ok=True)
    for name,regex in [('denoise_kernels',r'denoise_step_\d+'),('other_stages',r'text_encode|vae_decode')]:
        result=analyze(o/'trace.sqlite',regex,0)
        write(prof/(name+'.json'),result)
        (prof/(name+'.md')).write_text(render_md(result))
    mem=[]
    for number,line in enumerate((o/'tegrastats.log').read_text().splitlines(),1):
        m=re.search(r'RAM (\d+)/(\d+)MB .*?SWAP (\d+)/(\d+)MB',line)
        assert m, line
        a,b,d,e=map(int,m.groups())
        mem.append(dict(source_line=number,timestamp_local=line[:19],ram_reported_mb=a,ram_total_reported_mb=b,swap_reported_mb=d,ram_gib=a/1024,swap_gib=d/1024))
    with (r/'tegrastats-samples.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(mem[0]));w.writeheader();w.writerows(mem)
    log=(o/'sdcpp.log').read_text()
    segments={n:int(v) for n,v in re.findall(r'(qwen3|flux|vae) compute buffer size: .*?peak across (\d+) segments?',log)}
    report={'source':'originals/trace.sqlite and originals/results.jsonl','sqlite_integrity':'ok','nvtx_ranges':[{'name':n,'start_ns':a,'end_ns':b,'duration_ms':(b-a)/1e6} for a,b,n in ranges], 'gpu_activity_counts':{k:sum(t==k for t,a,b in activities) for k in sorted({t for t,a,b in activities})},'events_crossing_stage_ends':crossing,'segments_reported':segments,'monitor_window':{'samples':len(mem),'ram_peak_gib':max(x['ram_gib'] for x in mem),'ram_total_gib':mem[0]['ram_total_reported_mb']/1024,'swap_min_gib':min(x['swap_gib'] for x in mem),'swap_max_gib':max(x['swap_gib'] for x in mem)},'limitations':['One profiled generation; not baseline medians.','Stage assignment uses GPU start within CPU NVTX range; no events cross the reviewed stage ends.','Busy time is union of captured GPU activities, not SM utilization; its complement is not proven disk time.','System RAM samples include startup; no reliable wall-clock-to-stage mapping. Printed MB treated as MiB, consistent with existing Jetson metric convention.','GPU timestamps and CPU API time overlap; do not sum across those reports.']}
    write(r/'trace-review.json',report)
    # Lossless serialization of the harness RGB output, not model postprocessing.
    raw=(o/'raw/run-0.rgb').read_bytes();w,h,ch=g['width'],g['height'],g['channels']
    assert ch==3 and len(raw)==w*h*ch
    def chunk(tag,data):
        return struct.pack('!I',len(data))+tag+data+struct.pack('!I',zlib.crc32(tag+data)&0xffffffff)
    png=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!2I5B',w,h,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(b''.join(b'\0'+raw[y*w*3:(y+1)*w*3] for y in range(h))))+chunk(b'IEND',b'')
    (r/'output.png').write_bytes(png)
    summary=read('summary.json');summary['trace_timeline_review_pending']=False
    summary['unavailable_metrics']['gpu_stage_time']='Per-stage captured GPU activity available in profile/*.json; not a clean GPU timing benchmark.'
    summary['monitor_window']=report['monitor_window'];summary['segments_reported']=segments
    summary['profile_host_diagnostics']={'load_seconds':load['t_end']-load['t_start'],'generation_seconds':g['t_end']-g['t_start']}
    write(r/'summary.json',summary)
    write(r/'load.json',{'load_total_s':load['t_end']-load['t_start'],'scope':'Profiled new_sd_ctx; SHA256 pre-read warmed files'})
    write(r/'status.json',{'status':'complete','trace_timeline_review_pending':False,'evidence':'trace-review.json'})
    for name in ['runs.csv','stages.csv']:
        shutil.copyfile(o/name,r/name)
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
