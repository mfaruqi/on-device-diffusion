#!/usr/bin/env python3
"""Existing Jetson E2E, full-capture RAM and schematic weight residency; no upload."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    'jetson-torch-flux-klein-001__baseline__20261014-171410',
    'jetson-flux-klein-012__baseline__20261102-165643',
    'jetson-edgedit-flux-klein-006__baseline__20261014-170149',
    'jetson-nunchaku-flux-klein-001__baseline__20261102-120036',
]
LABELS = ['PyTorch eager', 'sd.cpp', 'edge-dit', 'Nunchaku']
COLORS = ['#5798d3', '#ed8737', '#70a844', '#ad8174']


def collect(names):
    records, hashes = [], {}
    def read(path):
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path.read_text()
    for label, color, name in zip(LABELS, COLORS, names):
        run = ROOT/'results/runs'/name
        config = json.loads(read(run/'config.json'))
        summary = json.loads(read(run/'summary.json'))
        assert json.loads(read(run/'status.json'))['status'] == 'complete', name
        w, opt = config['workload'], config['optimizations']
        assert w['height'] == w['width'] == 512
        assert w.get('steps', w.get('num_inference_steps')) == 4
        assert w.get('guidance_scale', w.get('cfg_scale', w.get('guidance'))) == 1
        assert not opt.get('conditioning_reuse') and not opt.get('step_cache')
        assert not opt.get('conditioning_cache_size')
        assert not config.get('engine_settings', {}).get('conditioning_cache_size')
        rows = [r for r in csv.DictReader(read(run/'runs.csv').splitlines()) if r['phase'] == 'measured']
        assert len(rows) == config['protocol']['measured_runs'] == 10, 'Requires completed full protocol'
        for field in ['wall_ms', 'text_encode_ms', 'denoise_ms', 'vae_decode_ms']:
            vals = [float(r[field]) for r in rows]
            for stat, v in [('median', statistics.median(vals)), ('min', min(vals)), ('max', max(vals))]:
                assert math.isclose(v, summary['measured'][field][stat], rel_tol=1e-8), (name, field, stat)
        rows.sort(key=lambda r: float(r['wall_ms']))
        middle = rows[4:6]
        values = {k: statistics.mean(float(r[k]) for r in middle)/1000
                  for k in ['wall_ms', 'text_encode_ms', 'denoise_ms', 'vae_decode_ms']}
        other = values['wall_ms']-sum(values[k] for k in ['text_encode_ms', 'denoise_ms', 'vae_decode_ms'])
        reload_s = 0
        policy = opt.get('stage_residency')
        if (run/'stage-residency.json').exists():
            residency = json.loads(read(run/'stage-residency.json'))
            policy = residency['policy']
            reload_s = statistics.mean(sum(residency['per_generation'][int(r['run_index'])][k]
                                           for k in ['transformer_free_s', 'text_encoder_load_s',
                                                     'text_encoder_free_s', 'transformer_load_s']) for r in middle)
            assert reload_s <= other+0.1
        samples = list(csv.DictReader(read(run/'tegrastats-samples.csv').splitlines()))
        interval_s = config['protocol']['tegrastats_interval_ms']/1000
        resident = label == 'edge-dit' or (label == 'sd.cpp' and config.get('engine_settings', {}).get('params_backend') == 'CUDA0')
        if label == 'sd.cpp' and not resident:
            assert opt.get('parameter_storage') == 'disk', 'Unrecognized sd.cpp residency policy'
        records.append(dict(label=label, color=color, source_run=name, config=config,
                            wall_s=values['wall_ms'], min_s=summary['measured']['wall_ms']['min']/1000,
                            max_s=summary['measured']['wall_ms']['max']/1000,
                            segments_s=[values['text_encode_ms'], values['denoise_ms'], values['vae_decode_ms'],
                                        reload_s, max(0, other-reload_s)],
                            sample_time_s=[(int(r['source_line'])-1)*interval_s for r in samples],
                            ram_gib=[float(r['ram_gib']) for r in samples],
                            swap_gib=[float(r['swap_gib']) for r in samples],
                            total_ram_gib=float(samples[0]['ram_total_reported_mb'])/1024,
                            memory_time_basis='nominal source-line index times configured sampling interval',
                            resident=resident, policy=policy, interval_s=interval_s))
    return records, hashes


def render(records, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42, 'svg.fonttype': 'none'}):
        fig, axes = plt.subplots(1, 3, figsize=(19, 7.6), gridspec_kw={'width_ratios': [1, 1.15, 1.25]})
        fig.subplots_adjust(left=.075, right=.985, top=.79, bottom=.30, wspace=.40)
        fig.suptitle('Jetson Orin Nano 8 GB · Klein Distilled · 512² · 4 steps', fontsize=21, fontweight='bold', y=.97)
        fig.text(.53, .90, 'Latency, system memory and weight residency · no conditioning reuse', ha='center', fontsize=13)
        a, b, c = axes
        shades = ['#70a9d1', '#7464a8', '#d5ad62', '#b3b3b3', '#e7e7e7']
        names = ['Text encoding', 'Denoising', 'VAE decode', 'Weight load/release¹', 'Other overhead']
        for i, r in enumerate(records):
            start = 0
            for value, shade, hatch in zip(r['segments_s'], shades, ['', '', '', '///', '']):
                a.barh(i, value, left=start, color=shade, edgecolor='white', height=.55, hatch=hatch, linewidth=.4)
                start += value
            a.errorbar(r['wall_s'], i, xerr=[[r['wall_s']-r['min_s']], [r['max_s']-r['wall_s']]], fmt='none', color='#333333', capsize=3)
            a.text(r['max_s']+1.2, i, f"{r['wall_s']:.2f} s", va='center', fontsize=10, fontweight='bold')
            b.plot(r['sample_time_s'], r['ram_gib'], color=r['color'], lw=1, label=r['label'])
        a.set_yticks(range(4)); a.set_yticklabels([r['label']+('²' if r['label']=='sd.cpp' else '') for r in records])
        a.invert_yaxis(); a.set_xlim(0, max(r['max_s'] for r in records)*1.25)
        a.set_xlabel('Generation latency (s)'); a.set_title('(a) End-to-end latency', fontweight='bold', pad=18)
        fig.legend(handles=[Patch(facecolor=sh, hatch='///' if i==3 else '', label=n) for i,(sh,n) in enumerate(zip(shades,names))],
                 loc='center', bbox_to_anchor=(.50,.205), ncol=5, fontsize=10, frameon=False)
        b.axhline(records[0]['total_ram_gib'], color='#333333', ls='--', lw=1, label='Physical RAM (7.44 GiB)')
        b.set_ylim(0,8); b.set_xlabel('Nominal time from capture start (s)')
        b.set_ylabel('Whole-system RAM used (GiB)'); b.set_title('(b) Memory over full capture', fontweight='bold', pad=18)
        b.legend(loc='lower right', fontsize=9, framealpha=.95)
        b.text(.02,.97,'Includes load, first image, warm-ups and 10 measured images',transform=b.transAxes,fontsize=8,va='top')
        # Stage columns communicate policy only; their equal widths do not encode durations.
        components = ['Encoder', 'DiT', 'VAE']
        for i, r in enumerate(records):
            for j, comp in enumerate(components):
                y = i*4+j
                # Resident means kept in GPU-visible memory, not continuously active.
                if r['resident']:
                    cells = [(0,3,'kept resident', '')]
                elif r['label'] in ['PyTorch eager', 'Nunchaku']:
                    cells = ([(0,1,'load / use', '///')] if j==0 else
                             [(1,2,'reload → keep', '///')] if j==1 else [(0,3,'kept resident','')])
                else:
                    cells = [(j,1,'load / use','///')]
                for x,width,text,hatch in cells:
                    c.barh(y,width,left=x,height=.68,color=r['color'],alpha=.85,edgecolor='white',hatch=hatch)
                    c.text(x+width/2,y,text,ha='center',va='center',fontsize=7.5,color='#111111')
        ticks = [i*4+j for i in range(4) for j in range(3)]
        c.set_yticks(ticks); c.set_yticklabels(components*4,fontsize=8)
        for i,r in enumerate(records):
            c.text(-.38,i*4+1,r['label'],ha='right',va='center',fontsize=10,fontweight='bold',rotation=90)
        c.set_ylim(14.8,-1); c.set_xlim(0,3)
        c.set_xticks([.5,1.5,2.5]); c.set_xticklabels(['Encode','Denoise','Decode'])
        c.set_xlabel('Pipeline stage (schematic; not to scale)')
        c.set_title('(c) Weight residency policy',fontweight='bold',pad=18)
        for x in [1,2]: c.axvline(x,color='#cccccc',ls=':',lw=.8)
        for ax in axes:
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        a.grid(axis='x',color='#dddddd',lw=.5); a.set_axisbelow(True)
        b.grid(color='#dddddd',lw=.5); b.set_axisbelow(True)
        notes = ['Latency: median and min–max of 10 measured images; stacks use the two middle observations ranked by total latency.',
                 '¹ PyTorch/Nunchaku load and release durations are measured separately. resident sd.cpp needs no per-image weight reload.',
                 ('² sd.cpp: completed resident + tiled run.' if records[1]['resident'] else
                  '² sd.cpp: previous completed disk-backed run; resident + tiled full run pending replacement. No ≈15 s estimate is plotted.'),
                 'RAM time uses sample index × interval (sd.cpp: 1 s; others: 0.2 s); no stage alignment is inferred. Residency is a policy schematic, not an allocation trace.',
                 'Mixed quantization/checkpoints and software differ; this is not a matched-quality ranking. Failed resident attempts are not in these completed-run traces.']
        fig.text(.075,.025,'\n'.join(notes),fontsize=9.3,linespacing=1.6,va='bottom',color='#444444')
        for ext in ['png','pdf','svg']:
            fig.savefig(out/f'jetson-latency-memory-residency.{ext}',dpi=180,facecolor='white')
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'output/jetson-residency')
    parser.add_argument('--sdcpp-run', default=RUNS[1], help='Completed run directory name under results/runs')
    args = parser.parse_args()
    names = list(RUNS); names[1] = args.sdcpp_run
    records, hashes = collect(names)
    args.out.mkdir(parents=True,exist_ok=True)
    render(records,args.out)
    (args.out/'figure-data.json').write_text(json.dumps(dict(records=records,source_sha256=hashes),indent=2)+'\n')
    print(args.out)


if __name__ == '__main__':
    main()
