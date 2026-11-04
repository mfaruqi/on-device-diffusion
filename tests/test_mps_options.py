import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from mps_options import COMPILE, TAYLOR, check_options, TaylorAudit


class MPSOptionTests(unittest.TestCase):
    def test_rejects_graph_break_fallback_and_combinations(self):
        check_options({'torch_compile': COMPILE})
        check_options({'step_cache': TAYLOR})
        for change in [{'fullgraph': False}, {'backend': 'eager'}, {'dynamic': True}]:
            cfg=copy.deepcopy(COMPILE);cfg.update(change)
            with self.assertRaises(AssertionError):check_options({'torch_compile':cfg})
        with self.assertRaises(AssertionError):check_options({'torch_compile':COMPILE,'step_cache':TAYLOR})

    def test_cache_audit_detects_missing_cfg_and_stale_state(self):
        audit=TaylorAudit.__new__(TaylorAudit)
        audit.names=['attention'];audit.generations=[]
        audit.events=[('attention',c,s,s<3) for c in ['cond','uncond'] for s in range(4)]
        audit.end(4,2)
        self.assertEqual(audit.generations[-1]['predict'],2)
        audit.events=audit.events[:4]
        with self.assertRaises(AssertionError):audit.end(4,2)
        audit.events=[('attention','cond',s+4,s<3) for s in range(4)]
        with self.assertRaises(AssertionError):audit.end(4,1)
