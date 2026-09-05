"""
KV Cache utilities for R-CCR three-path inference.

Handles both tuple-format and DynamicCache-format past_key_values.
Provides stack/split operations for batching deep+shallow decode passes.
"""

import torch
from typing import Tuple, Any, Sequence


# ── Type detection ──────────────────────────────────────────────

def _is_dynamic_cache(cache) -> bool:
    return hasattr(cache, "key_cache") and hasattr(cache, "value_cache")


# ── Stack: merge two batch-1 caches into one batch-2 cache ─────

def stack_kv_caches(cache_a, cache_b):
    """
    Stack two KV caches (each batch=1) along the batch dimension → batch=2.
    Both caches MUST have the same sequence length.
    """
    if _is_dynamic_cache(cache_a):
        return _stack_dynamic(cache_a, cache_b)
    else:
        return _stack_tuple(cache_a, cache_b)


def _stack_dynamic(ca, cb):
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        raise TypeError("DynamicCache not available")

    out = DynamicCache()
    out.key_cache = [
        torch.cat([ca.key_cache[i], cb.key_cache[i]], dim=0)
        for i in range(len(ca.key_cache))
    ]
    out.value_cache = [
        torch.cat([ca.value_cache[i], cb.value_cache[i]], dim=0)
        for i in range(len(ca.value_cache))
    ]
    if hasattr(ca, "_seen_tokens"):
        out._seen_tokens = ca._seen_tokens
    return out


def _stack_tuple(ca, cb):
    return tuple(
        (torch.cat([ka, kb], dim=0), torch.cat([va, vb], dim=0))
        for (ka, va), (kb, vb) in zip(ca, cb)
    )


# ── Split: batch-2 cache → two batch-1 caches ─────────────────

def split_kv_caches(cache) -> Tuple[Any, Any]:
    """Split a batch-2 KV cache back into two batch-1 caches."""
    if _is_dynamic_cache(cache):
        return _split_dynamic(cache)
    else:
        return _split_tuple(cache)


def cache_seq_length(cache) -> int:
    """Return the cached sequence length for tuple or DynamicCache formats."""
    if _is_dynamic_cache(cache):
        return int(cache.key_cache[0].shape[-2])
    return int(cache[0][0].shape[-2])


def pad_kv_cache_right(cache, pad_tokens: int):
    """Right-pad the sequence dimension of every KV tensor with zeros."""
    if pad_tokens <= 0:
        return cache
    if _is_dynamic_cache(cache):
        try:
            from transformers.cache_utils import DynamicCache
        except ImportError:
            raise TypeError("DynamicCache not available")
        out = DynamicCache()
        out.key_cache = [
            torch.nn.functional.pad(k, (0, 0, 0, pad_tokens))
            for k in cache.key_cache
        ]
        out.value_cache = [
            torch.nn.functional.pad(v, (0, 0, 0, pad_tokens))
            for v in cache.value_cache
        ]
        if hasattr(cache, "_seen_tokens"):
            out._seen_tokens = cache._seen_tokens + pad_tokens
        return out
    return tuple(
        (
            torch.nn.functional.pad(k, (0, 0, 0, pad_tokens)),
            torch.nn.functional.pad(v, (0, 0, 0, pad_tokens)),
        )
        for k, v in cache
    )


def stack_kv_caches_many(caches: Sequence[Any]):
    """Stack any number of equal-length, batch-1 KV caches."""
    if not caches:
        raise ValueError("At least one KV cache is required")
    lengths = [cache_seq_length(cache) for cache in caches]
    if len(set(lengths)) != 1:
        raise ValueError("KV cache lengths differ: %r" % lengths)
    if _is_dynamic_cache(caches[0]):
        try:
            from transformers.cache_utils import DynamicCache
        except ImportError:
            raise TypeError("DynamicCache not available")
        out = DynamicCache()
        out.key_cache = [
            torch.cat([cache.key_cache[layer] for cache in caches], dim=0)
            for layer in range(len(caches[0].key_cache))
        ]
        out.value_cache = [
            torch.cat([cache.value_cache[layer] for cache in caches], dim=0)
            for layer in range(len(caches[0].value_cache))
        ]
        if hasattr(caches[0], "_seen_tokens"):
            out._seen_tokens = lengths[0]
        return out
    return tuple(
        (
            torch.cat([cache[layer][0] for cache in caches], dim=0),
            torch.cat([cache[layer][1] for cache in caches], dim=0),
        )
        for layer in range(len(caches[0]))
    )


def split_kv_caches_many(cache, count: int):
    """Split a batched KV cache into ``count`` batch-1 caches."""
    if count <= 0:
        raise ValueError("count must be positive")
    if _is_dynamic_cache(cache):
        try:
            from transformers.cache_utils import DynamicCache
        except ImportError:
            raise TypeError("DynamicCache not available")
        outputs = []
        for index in range(count):
            item = DynamicCache()
            item.key_cache = [k[index:index + 1] for k in cache.key_cache]
            item.value_cache = [v[index:index + 1] for v in cache.value_cache]
            if hasattr(cache, "_seen_tokens"):
                item._seen_tokens = cache._seen_tokens
            outputs.append(item)
        return tuple(outputs)
    return tuple(
        tuple((k[index:index + 1], v[index:index + 1]) for k, v in cache)
        for index in range(count)
    )


def _split_dynamic(cache):
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        raise TypeError("DynamicCache not available")

    ca, cb = DynamicCache(), DynamicCache()
    ca.key_cache   = [k[:1] for k in cache.key_cache]
    ca.value_cache = [v[:1] for v in cache.value_cache]
    cb.key_cache   = [k[1:2] for k in cache.key_cache]
    cb.value_cache = [v[1:2] for v in cache.value_cache]
    if hasattr(cache, "_seen_tokens"):
        ca._seen_tokens = cache._seen_tokens
        cb._seen_tokens = cache._seen_tokens
    return ca, cb


def _split_tuple(cache):
    ca = tuple((k[:1], v[:1]) for k, v in cache)
    cb = tuple((k[1:2], v[1:2]) for k, v in cache)
    return ca, cb


# ── Free cache memory ─────────────────────────────────────────

def free_cache(cache):
    """Explicitly free GPU memory held by a KV cache."""
    if cache is None:
        return
    if _is_dynamic_cache(cache):
        cache.key_cache.clear()
        cache.value_cache.clear()
    elif isinstance(cache, (tuple, list)):
        del cache
