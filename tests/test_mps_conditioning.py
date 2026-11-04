import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from mps_conditioning import ConditioningCache


class ConditioningTests(unittest.TestCase):
    def test_positive_negative_and_encoding_settings_are_distinct(self):
        calls = []
        def encode(prompt, prompt_embeds=None, max_sequence_length=512):
            calls.append((prompt, max_sequence_length))
            return object(), object()
        pipe = SimpleNamespace(encode_prompt=encode)
        cache = ConditioningCache(pipe, 2)
        positive = pipe.encode_prompt('cat')
        negative = pipe.encode_prompt('')
        self.assertIs(pipe.encode_prompt(prompt='cat', max_sequence_length=512), positive)
        self.assertIs(pipe.encode_prompt(''), negative)
        self.assertEqual(calls, [('cat', 512), ('', 512)])
        self.assertEqual((cache.hits, cache.misses), (2, 2))
        with self.assertRaises(AssertionError):
            pipe.encode_prompt('cat', max_sequence_length=256)
        with self.assertRaises(AssertionError):
            pipe.encode_prompt('cat', prompt_embeds=object())

    def test_separate_pipeline_never_reuses_other_models_embeddings(self):
        make = lambda: SimpleNamespace(encode_prompt=lambda prompt: object())
        a, b = make(), make()
        ConditioningCache(a, 1)
        ConditioningCache(b, 1)
        self.assertIsNot(a.encode_prompt('cat'), b.encode_prompt('cat'))


if __name__ == '__main__':
    unittest.main()
