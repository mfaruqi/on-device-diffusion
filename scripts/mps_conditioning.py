"""Exact, per-pipeline prompt encoding reuse for fixed-workload MPS experiments."""
import inspect


class ConditioningCache:
    def __init__(self, pipe, capacity):
        self.original = pipe.encode_prompt
        self.signature = inspect.signature(self.original)
        self.capacity = capacity
        self.values = {}
        self.hits = 0
        self.misses = 0
        pipe.encode_prompt = self.encode

    def encode(self, *args, **kwargs):
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        assert bound.arguments.get('prompt_embeds') is None, 'Precomputed embeddings are outside this policy'
        # Include every encoding argument; the cache belongs to one immutable pipeline.
        key = repr(bound.arguments)
        if key in self.values:
            self.hits += 1
            return self.values[key]
        assert len(self.values) < self.capacity, 'Conditioning changed beyond the declared fixed workload'
        result = self.original(*args, **kwargs)
        self.values[key] = result
        self.misses += 1
        return result

    def receipt(self):
        return dict(policy='exact_prompt_encoding', hits=self.hits, misses=self.misses,
                    entries=len(self.values), capacity=self.capacity,
                    scope='One loaded pipeline; identical encoding arguments; positive and CFG negative prompts separate')
