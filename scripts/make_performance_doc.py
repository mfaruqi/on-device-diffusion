#!/usr/bin/env python3
"""Write docs/performance-a100-jetson.md: benchmark tables in the layout of edge-dit.cpp's docs/performance-4090.md.

  python scripts/make_performance_doc.py

Every number is read from run files, never typed: the run catalogue and labels come from
output/stage-breakdown-20261007/chart-data.json (completed, validated full protocols only); latency medians,
memory and run counts from each run's summary.json; GenEval from results/geneval (via export_quality_wandb.py);
paired PSNR/LPIPS from each run's quality-diagnostics.json; speedups from output/speedups-20261028/speedups.json
when present (pairs defined in scripts/plot_speedups.py). Re-run after new runs are added to the chart data.
"""
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
RUNS = ROOT/'results'/'runs'
CHART = ROOT/'output'/'stage-breakdown-20261007'/'chart-data.json'
SPEEDUPS = ROOT/'output'/'speedups-20261028'/'speedups.json'
OUT = ROOT/'docs'/'performance-a100-jetson.md'
DEVICES = {'A100': ('A100-PCIE-40GB', 'A100 PCIe 40GB (Gilbreth)', '1024x1024'),
           'Jetson': ('Jetson', 'Jetson Orin Nano 8GB (25 W)', '512x512')}
SYSTEM_ORDER = ['PyTorch / Diffusers', 'stable-diffusion.cpp', 'edge-dit.cpp', 'Nunchaku']
DEGRADED = ('degraded', 'garbled', 'deformed', 'output changed')


def system(engine):
    return next((s for s in SYSTEM_ORDER if engine.startswith(s.split(' ')[0])), engine)


def med(summary, key):
    v = summary['measured'].get(key)
    return v['median'] if isinstance(v, dict) and v.get('median') is not None else None


def memory(summary, device):
    """A100: device-wide used peak (GiB). Jetson: whole-harness RAM peak incl. load (shared memory, GiB)."""
    if device == 'A100':
        v = med(summary, 'device_used_peak_gib') or summary.get('device_used_peak_gib')
        return (v['median'] if isinstance(v, dict) else v), 'device used'
    s = summary.get('system_memory') or summary.get('monitor_window') or {}
    return s.get('ram_peak_gib'), 'RAM incl. load'


def ms(v):
    return '—' if v is None else f'{v:.0f}' if v >= 100 else f'{v:.1f}'


def reference_runs():
    if not SPEEDUPS.exists():
        return set()
    return {r['reference_run'] for r in json.loads(SPEEDUPS.read_text())['rows'] if 'reference_run' in r}


def rows_for(device, geneval):
    gpu = DEVICES[device][0]
    refs = reference_runs()
    out = []
    for r in json.loads(CHART.read_text())['runs']:
        if not r['gpu'].startswith(gpu):
            continue
        s = json.loads((RUNS/r['run_dir']/'summary.json').read_text())
        parts = r['workload'].split('; ')
        model = 'klein-base' if parts[0].startswith('Base') else 'klein-distilled'
        mem, basis = memory(s, device)
        label = r['label'].replace('<br>', ' ')
        q = RUNS/r['run_dir']/'quality-diagnostics.json'
        g = geneval.get(s.get('id') or r['config']['id'], {})
        worse = g.get('quality/geneval_delta_ci95_high', 1) < 0  # paired GenEval interval entirely below zero
        out.append({'run': r['run_dir'], 'config_id': s.get('id') or r['config']['id'], 'model': model,
                    'workload': '; '.join(parts[:4]), 'system': system(r['engine']), 'precision': parts[4],
                    'budget': ('; '.join(parts[5:]) or '—').split('. ')[0], 'label': label,
                    'reference': r['run_dir'] in refs or 'reference' in label.lower(), 'n': s.get('measured_runs'),
                    'dit': med(s, 'denoise_ms'), 'e2e': med(s, 'wall_ms'), 'te': med(s, 'text_encode_ms'),
                    'vae': med(s, 'vae_decode_ms'), 'mem': mem, 'mem_basis': basis,
                    'degraded': any(w in label.lower() for w in DEGRADED) or worse,
                    'diag': json.loads(q.read_text()) if q.exists() else None})
    out.sort(key=lambda x: (x['model'] != 'klein-distilled', SYSTEM_ORDER.index(x['system'])
                            if x['system'] in SYSTEM_ORDER else 9, not x['reference'], x['e2e']))
    return out


def speed_table(rows):
    best = {id(min((r for r in rows if not r['degraded']), key=lambda r: r['e2e']))}
    lines = ['| System | Precision | Budget | Configuration | DiT | E2E | TE | VAE | n |',
             '|---|---|---|---|--:|--:|--:|--:|--:|']
    for r in rows:
        b = '**' if id(r) in best else ''
        flag = ' ⚠' if r['degraded'] else ''
        lines.append(f"| {b}{r['system']}{b} | {r['precision']} | {r['budget']} | {b}{r['label']}{flag}{b} | "
                     f"{ms(r['dit'])} | {b}{ms(r['e2e'])}{b} | {ms(r['te'])} | {ms(r['vae'])} | {r['n']} |")
    return lines


def memory_table(rows):
    lines = ['| Model | System | Precision | Budget | Configuration | Peak (GiB) | Basis |', '|---|---|---|---|---|--:|---|']
    have = [r for r in rows if r['mem'] is not None]
    best = {id(min((r for r in have if r['model'] == m), key=lambda r: r['mem'])) for m in {r['model'] for r in have}}
    for r in have:
        mem = f"**{r['mem']:.2f}**" if id(r) in best else f"{r['mem']:.2f}"
        lines.append(f"| {r['model']} | {r['system']} | {r['precision']} | {r['budget']} | {r['label']} | {mem} | {r['mem_basis']} |")
    return lines


def quality_table(rows, geneval):
    lines = ['| Model | System | Configuration | GenEval | Δ vs same-system ref [95% CI] | prompts | PSNR (dB) | LPIPS |',
             '|---|---|---|--:|---|--:|--:|--:|']
    scored = [r for r in rows if r['config_id'] in geneval]
    best = set()
    for m in {r['model'] for r in scored}:  # compare only scores from the same prompt count
        same = [r for r in scored if r['model'] == m]
        n = max(geneval[r['config_id']]['quality/geneval_prompts'] for r in same)
        best.add(id(max((r for r in same if geneval[r['config_id']]['quality/geneval_prompts'] == n),
                        key=lambda r: geneval[r['config_id']]['quality/geneval'])))
    for r in rows:
        g = geneval.get(r['config_id'])
        d = r['diag']
        if not g and not d:
            continue
        if g:
            score = f"{g['quality/geneval']:.3f}"
            score = f'**{score}**' if id(r) in best else score
            delta = (f"{g['quality/geneval_delta']:+.3f} [{g['quality/geneval_delta_ci95_low']:+.3f}, "
                     f"{g['quality/geneval_delta_ci95_high']:+.3f}]" if 'quality/geneval_delta' in g else '— (reference)')
            prompts = g['quality/geneval_prompts']
        else:
            score, delta, prompts = '—', '—', '—'
        psnr = d['psnr_db'] if d else None
        psnr = '∞ (identical)' if isinstance(psnr, str) else ('—' if psnr is None else f'{psnr:.1f}')
        lpips = '—' if not d else f"{d['lpips_alex_v0_1']:.3f}"
        lines.append(f"| {r['model']} | {r['system']} | {r['label']}{' ⚠' if r['degraded'] else ''} | {score} | {delta} | {prompts} | {psnr} | {lpips} |")
    return lines


def speedup_table(device, degraded):
    if not SPEEDUPS.exists():
        return ['(speedup pairs not generated; run scripts/plot_speedups.py)']
    rows = [r for r in json.loads(SPEEDUPS.read_text())['rows'] if r['group'].startswith(device) and 'speedup' in r]
    lines = ['| Workload | System | Option | Reference E2E (s) | Option E2E (s) | Speedup |', '|---|---|---|--:|--:|--:|']
    for r in rows:
        r['degraded_quality'] = r['run'] in degraded  # same ⚠ rule as the speed tables
    top = max((r for r in rows if not r['degraded_quality']), key=lambda r: r['speedup'])
    for r in rows:
        lines.append(f"| {r['group'].removeprefix(device + ' ')} | {r['panel'][4:]} | {r['bar']}{' ⚠' if r['degraded_quality'] else ''}"
                     f"{' †' if r['incremental'] else ''} | {r['reference_median_s']:.2f} | {r['variant_median_s']:.2f} | "
                     f"{'**' if r is top else ''}{r['speedup']:.2f}×{'**' if r is top else ''} |")
    return lines


def readout(rows):
    lines = ['| Model | System | Precision | Budget | Reference | DiT | E2E | Peak (GiB) |', '|---|---|---|---|---|--:|--:|--:|']
    refs = [r for r in rows if r['reference']]
    best = {id(min((r for r in refs if r['model'] == m), key=lambda r: r['e2e'])) for m in {r['model'] for r in refs}}
    for r in refs:
        mem = '—' if r['mem'] is None else f"{r['mem']:.2f}"
        e2e = f"**{ms(r['e2e'])}**" if id(r) in best else ms(r['e2e'])
        lines.append(f"| {r['model']} | {r['system']} | {r['precision']} | {r['budget']} | {r['label']} | "
                     f"{ms(r['dit'])} | {e2e} | {mem} |")
    return lines


def completeness(device):
    prefix = 'a100-' if device == 'A100' else 'jetson-'
    c = Counter()
    for status in RUNS.glob(f'{prefix}*/status.json'):
        c[json.loads(status.read_text()).get('status', 'unknown')] += 1
    return c


def main():
    import export_quality_wandb as quality
    geneval = quality.attachments()
    md = ['# Performance and Benchmarks (A100 PCIe 40GB, Jetson Orin Nano)', '',
          '[← Project wiki](../wiki/hot.md) · [metric definitions](../wiki/methods/baseline-metrics.md) · '
          '[experiment registry](../wiki/experiments.md)', '',
          'Benchmark snapshot for FLUX.2 klein 4B text-to-image on two devices, laid out after edge-dit.cpp\'s '
          '[performance (RTX 4090)](https://github.com/THU-MIG/edge-dit.cpp/blob/main/docs/performance-4090.md) page. '
          'It reports speed, peak memory and image quality for three runtimes (PyTorch/Diffusers, stable-diffusion.cpp, '
          'edge-dit.cpp; Nunchaku on the A100) and their options. Each device section stands alone; cross-device '
          'conclusions are in the comparison records linked at the end. Generated by `scripts/make_performance_doc.py` '
          'from run files; do not edit the tables by hand.', '',
          '## How to read these numbers', '',
          '- **Compare speed with `DiT`** (median denoising-stage time, ms) for runtime efficiency, and **`E2E`** '
          '(median `wall_ms`: one generation from prompt to decoded image) for what a user waits. '
          '**Model load is excluded** from both (reported in each run\'s `load.json`), but on the Jetson some '
          'configurations reload weights inside every generation by design (sequential residency, disk-backed '
          'sd.cpp); that reload is inside `E2E` and is the main difference between those rows.',
          '- **Medians over `n` measured generations** after one first generation and discarded warm-ups; '
          'one prompt, seed 0, batch 1. TE = text encoding, VAE = VAE decoding (ms).',
          '- **Same-precision rule:** compare systems only within the same precision column. On the A100 the 16-bit '
          'group is PyTorch BF16 / sd.cpp BF16 / edge-dit BF16 weights with FP32 activations; edge-dit Base Q4_0 and '
          'Nunchaku W4A4 are compressed checkpoints and are reported separately, never against BF16. On the Jetson '
          'all transformers are Q4_0, but text encoders differ (sd.cpp Q4_K_M, edge-dit Q4_K_M with Q8_0 token '
          'embedding, PyTorch NF4).',
          '- **⚠** = the option changes the image in a way measured as worse: a paired GenEval interval entirely below '
          'zero, or visibly damaged text/content in the benchmark image. **Bold** = the best result in each table '
          '(speed: fastest E2E row without ⚠; readout: fastest reference per model; memory: lowest peak per model; '
          'quality: highest GenEval per model among rows scored on the largest prompt set; options: largest speedup without ⚠). Speed bold spans precision '
          'groups, so read it together with the same-precision rule.',
          '- **Quality:** GenEval is the formal metric (calibration prompts, 4 seeds; Δ is paired against the same '
          'system\'s reference). PSNR/LPIPS compare the single benchmark image with the same system\'s reference '
          'image (∞ = identical pixels). Jetson GenEval rows are A100 generations of the identical Jetson weights and '
          'options (quality depends on weights, precision and options, not the device). See [GenEval method](../wiki/methods/baseline-metrics.md#geneval-formal-image-quality-metric).',
          '- **Memory is in GiB.** A100: device-wide used memory peak. Jetson: whole-harness RAM peak including '
          'load (CPU and GPU share 7.4 GiB). The two are different quantities and are not compared.', '']
    md += ['## Test matrix at a glance', '', '| Device | Workloads | Systems | Runs on disk (complete/failed) |',
           '|---|---|---|---|']
    for device, (_, name, res) in DEVICES.items():
        rows = rows_for(device, geneval)
        c = completeness(device)
        md.append(f"| {name} | klein distilled (4 steps) / Base (50 steps, CFG 4), {res} | "
                  f"{', '.join(sorted({r['system'] for r in rows}, key=SYSTEM_ORDER.index))} | "
                  f"{sum(c.values())} ({c['complete']}/{c['failed']}) |")
    md += ['', 'Failed runs include memory-guard stops, out-of-memory placements, smoke tests that failed a gate and '
           'interrupted runs; each keeps its config and error in `status.json` and is listed in the '
           '[registry](../wiki/experiments.md). Only completed full protocols appear in the tables.', '',
           '## Speed-up overview', '',
           'Each option against its own reference on the same device, workload and system (log scale; hatched = '
           'output measurably worse; generated by `scripts/plot_speedups.py`, copied to `docs/assets/`). Per-device tables follow.', '',
           '![Speedups per device and workload](assets/speedups-20261028.png)', '']
    for device, (_, name, res) in DEVICES.items():
        rows = rows_for(device, geneval)
        md += ['---', '', f'## {name} — text-to-image ({res})', '']
        for model, title in (('klein-distilled', 'klein distilled (4 steps, CFG 1)'), ('klein-base', 'klein Base (50 steps, CFG 4)')):
            sub = [r for r in rows if r['model'] == model]
            md += [f'### Speed: {title} (median, ms)', '', *speed_table(sub), '']
        md += ['### Same-precision cross-system readout', '',
               'Each system\'s reference rows (numbers only; matching and interpretation are in the comparison '
               'records under [Related documentation](#related-documentation)). Compare only rows with the same precision.',
               '', *readout(rows), '']
        md += ['### Memory (peak, GiB)', '', *memory_table(rows), '']
        md += ['### Quality (GenEval; PSNR/LPIPS vs same-system reference image)', '', *quality_table(rows, geneval), '']
        md += ['### Options against their own reference', '',
               'Speedup = reference E2E median / option E2E median, same system and workload. † measured on top of '
               'conditioning reuse (its labelled reference).', '', *speedup_table(device, {r['run'] for r in rows if r['degraded']}), '']
    md += ['---', '', '## Reproducibility', '',
           '- **A100:** Gilbreth `a100-40gb` partition, PCIe A100-40GB nodes (`--constraint=G`), one GPU per job, '
           'offline model cache, revisions pinned in each `*.resolved.json`. Runners: `scripts/run_flux_portable.py` '
           '(PyTorch), `scripts/run_sdcpp.py`, `scripts/run_edgedit.py`, `scripts/run_nunchaku.py`.',
           '- **Jetson:** Orin Nano 8GB, JetPack 6, 25 W power mode, desktop session closed for long runs, memory '
           'guard enabled; procedure in [Jetson how-to](../wiki/methods/jetson-howto.md).',
           '- **Protocol:** one first generation (reported separately in each record), warm-ups discarded, '
           '`n` measured generations; medians and min–max in each `summary.json`.',
           '- **Regenerate:** `python3 scripts/make_performance_doc.py` (reads `results/runs/*/summary.json`, '
           '`results/geneval/`, `output/stage-breakdown-20261007/chart-data.json`).', '',
           '## Related documentation', '',
           '- Cross-device and cross-engine comparisons: '
           '[Jetson engines, matched options](../experiments/compare-jetson-engines-matched-options.md), '
           '[A100 PyTorch vs sd.cpp](../experiments/compare-a100-pytorch-vs-sdcpp-flux-klein.md), '
           '[Jetson Base caches, reuse, compile](../experiments/compare-jetson-base-caches-reuse-compile.md).',
           '- Quality: [GenEval A100 distilled](../experiments/compare-geneval-a100-distilled-20261024.md), '
           '[GenEval A100 Base](../experiments/compare-geneval-a100-base-20261025.md), '
           '[GenEval Jetson-matched](../experiments/compare-geneval-jetson-matched-20261026.md), '
           '[GenEval edge-dit Base](../experiments/compare-geneval-a100-edgedit-base-20261027.md).',
           '- Stage charts: [Klein stage breakdown](../output/stage-breakdown-20261007/README.md).', '']
    OUT.write_text('\n'.join(md))
    print('wrote', OUT.relative_to(ROOT))


if __name__ == '__main__':
    main()
