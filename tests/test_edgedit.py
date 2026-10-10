"""Guard against accepting incomplete protocols or relabelling engine clocks."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_edgedit import check_config, parse_log
from measurement import summarize_runs

ROOT=Path(__file__).resolve().parents[1]


class EdgeSampleTests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads((ROOT/'configs/a100-edgedit-flux-klein-bf16-adapter-smoke.json').read_text())
        self.phases=check_config(self.cfg)
        confirmations=['default backend: CUDA0','flux activation dtype: f32 (bf16_tensors=149/149)',
                       'Conditioner weight type stat: bf16:398','Diffusion model weight type stat: bf16:149',
                       'VAE weight type stat: bf16:248','cache mode   : original']
        self.log='\n'.join(confirmations)+'\n'
        for i in range(2):
            self.log+='1024x1024 latent=64x64 image_seq_len=4096 steps=4 flux2_mu=2.291 guidance=1.00 cfg=1.00 seed=0\n'
            for j,(stage,event) in enumerate([('encode','begin'),('encode','end'),('denoise','begin'),('denoise','end'),('decode','begin'),('decode','end')]):
                self.log+=f'[[phase]] stage={stage} event={event} t={100+i*2+j*.2:.6f}\n'
            for step,(a,b) in enumerate([('1.000000','0.967384'),('0.967384','0.908144'),('0.908144','0.767200'),('0.767200','0.000000')],1):
                self.log+=f'flux step {step}/4 sigma={a} next={b}\n'
            self.log+=f'[ed-sample] pass {i+1}/2  1/1  seed=0  1.001s\n'
        self.timing={'num_images':2,'e2e_time':{'total':2.002}}

    def parse(self,log):
        return parse_log(log,self.cfg,self.phases,self.timing)

    def test_sources_remain_distinct_and_unknowns_stay_empty(self):
        rows,stages,events=self.parse(self.log)
        self.assertIsNone(rows[1]['text_encode_ms'])
        self.assertIsNone(rows[0]['image_sha256'])
        self.assertEqual(stages[0]['stage'],'encode_setup')
        self.assertIsNone(stages[0]['gpu_ms'])
        self.assertEqual(events[1]['clock'],'edgedit_system_clock')
        self.assertIsNone(events[0]['start_s'])
        self.assertEqual(summarize_runs(rows)['measured']['wall_ms']['n'],1)

    def test_incomplete_stage_or_schedule_rejected(self):
        for log in [self.log.replace('stage=decode event=end','stage=decode event=begin',1),
                    self.log.replace('sigma=1.000000 next=0.967384','sigma=1.000000 next=0.900000',1),
                    self.log.replace('pass 2/2','pass 3/2',1)]:
            with self.subTest(log=log[-50:]),self.assertRaises(AssertionError):self.parse(log)

    def test_clock_jump_rejected(self):
        with self.assertRaises(AssertionError):
            self.parse(self.log.replace('t=101.000000','t=109.000000'))

    def test_unverified_cache_rejected(self):
        self.cfg['optimizations']['step_cache']={'mode':'dicache'}
        with self.assertRaises(AssertionError):check_config(self.cfg)


if __name__=='__main__':unittest.main()
