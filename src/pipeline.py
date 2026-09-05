"""
R-CCR Generation Pipeline

Two modes:
  Cached   (default) : prefill once → decode with KV cache  (fast)
  Uncached (fallback): full three-path forward every step    (slow)
"""

import torch
import time
from typing import Optional, Dict
from PIL import Image

from .decoder.rccr_decoder import RCCRDecoder
from .models.base_adapter import BaseModelAdapter


class RCCRPipeline:
    """
    Risk-Calibrated Cross-modal Conflict Resolution pipeline.

    Automatically uses KV-cache mode when the adapter supports it.
    """

    def __init__(
        self,
        model_adapter: BaseModelAdapter,
        lambda_l: float = 1.0,
        lambda_s: float = 0.5,
        tau_l: float = 0.3,
        tau_v: float = 0.3,
        top_k_proxy: int = 10,
        theta_apc: float = 0.30,
        apc_mode: str = "relative_max",
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        log_dir: Optional[str] = None,
        preserve_binary_tokens_first_step: bool = False,
        binary_token_variants=None,
        binary_margin_threshold: float = 0.0,
    ):
        self.adapter = model_adapter
        self.decoder = RCCRDecoder(
            lambda_l=lambda_l, lambda_s=lambda_s,
            tau_l=tau_l, tau_v=tau_v,
            top_k_proxy=top_k_proxy, theta_apc=theta_apc,
            apc_mode=apc_mode, do_sample=do_sample,
            temperature=temperature, top_p=top_p, top_k=top_k,
            log_dir=log_dir,
            preserve_binary_tokens_first_step=preserve_binary_tokens_first_step,
            binary_token_variants=binary_token_variants,
            binary_margin_threshold=binary_margin_threshold,
        )

    def generate(self, image: Image.Image, prompt: str,
                 max_new_tokens: int = 512, sample_id: str = "") -> Dict:
        if self.adapter.supports_kv_cache():
            return self._generate_cached(image, prompt, max_new_tokens, sample_id)
        return self._generate_uncached(image, prompt, max_new_tokens, sample_id)

    # ──────────────────────────────────────────────────────────
    # Cached path  (prefill once → incremental decode)
    # ──────────────────────────────────────────────────────────
    def _generate_cached(self, image, prompt, max_new_tokens, sample_id):
        tokenizer = self.adapter.tokenizer
        eos_ids = set(self.adapter.get_eos_token_ids())
        generated = []
        changed = 0
        safety_fallbacks = 0
        first_step_binary = {}
        t0 = time.time()

        # ── prefill (runs 3 full forward passes, once) ──
        ld, ls, lb, cache = self.adapter.prefill(image, prompt)
        min_new_tokens = max(
            int(getattr(self.adapter, "min_new_tokens", 0)), 0
        )

        for step in range(max_new_tokens):
            if step < min_new_tokens:
                ld = ld.clone()
                ls = ls.clone()
                lb = lb.clone()
                for eos in eos_ids:
                    ld[eos] = float("-inf")
                    ls[eos] = float("-inf")
                    lb[eos] = float("-inf")
            res = self.decoder.decode_step(
                logits_deep=ld, logits_shallow=ls, logits_blind=lb,
                step_id=step, sample_id=sample_id, tokenizer=tokenizer,
            )
            if step == 0:
                first_step_binary = {
                    key: res.get(key)
                    for key in (
                        "binary_margin_threshold",
                        "binary_margin_before_threshold",
                        "binary_margin_after_threshold",
                        "binary_yes_token_id",
                        "binary_no_token_id",
                    )
                }
            tid = res["selected_token_id"]
            resolver = getattr(self.adapter, "resolve_degenerate_token", None)
            if resolver is not None:
                tid, used_fallback = resolver(ld, generated, tid, eos_ids)
                safety_fallbacks += int(used_fallback)
            if tid in eos_ids:
                break
            generated.append(tid)
            if res["changed"]:
                changed += 1

            # ── incremental decode (1-2 tiny forward calls) ──
            if step < max_new_tokens - 1:
                ld, ls, lb, cache = self.adapter.decode_step_cached(tid, cache)

        elapsed = time.time() - t0
        text = tokenizer.decode(generated, skip_special_tokens=True)
        prefix = getattr(self.adapter, "response_prefix", "")
        if prefix:
            text = (prefix + " " + text).strip()
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            generated = prefix_ids + generated
        return self._pack(
            text, generated, changed, elapsed,
            safety_fallbacks=safety_fallbacks,
            first_step_binary=first_step_binary,
        )

    # ──────────────────────────────────────────────────────────
    # Uncached path  (full re-encode every step — legacy)
    # ──────────────────────────────────────────────────────────
    def _generate_uncached(self, image, prompt, max_new_tokens, sample_id):
        tokenizer = self.adapter.tokenizer
        eos_ids = set(self.adapter.get_eos_token_ids())
        generated = []
        changed = 0
        safety_fallbacks = 0
        first_step_binary = {}
        t0 = time.time()
        min_new_tokens = max(
            int(getattr(self.adapter, "min_new_tokens", 0)), 0
        )

        for step in range(max_new_tokens):
            ld, ls, lb = self.adapter.get_three_path_logits(
                image, prompt, generated if generated else None)
            if step < min_new_tokens:
                ld = ld.clone()
                ls = ls.clone()
                lb = lb.clone()
                for eos in eos_ids:
                    ld[eos] = float("-inf")
                    ls[eos] = float("-inf")
                    lb[eos] = float("-inf")
            res = self.decoder.decode_step(
                logits_deep=ld, logits_shallow=ls, logits_blind=lb,
                step_id=step, sample_id=sample_id, tokenizer=tokenizer,
            )
            if step == 0:
                first_step_binary = {
                    key: res.get(key)
                    for key in (
                        "binary_margin_threshold",
                        "binary_margin_before_threshold",
                        "binary_margin_after_threshold",
                        "binary_yes_token_id",
                        "binary_no_token_id",
                    )
                }
            tid = res["selected_token_id"]
            resolver = getattr(self.adapter, "resolve_degenerate_token", None)
            if resolver is not None:
                tid, used_fallback = resolver(ld, generated, tid, eos_ids)
                safety_fallbacks += int(used_fallback)
            if tid in eos_ids:
                break
            generated.append(tid)
            if res["changed"]:
                changed += 1

        elapsed = time.time() - t0
        text = tokenizer.decode(generated, skip_special_tokens=True)
        prefix = getattr(self.adapter, "response_prefix", "")
        if prefix:
            text = (prefix + " " + text).strip()
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            generated = prefix_ids + generated
        return self._pack(
            text, generated, changed, elapsed,
            safety_fallbacks=safety_fallbacks,
            first_step_binary=first_step_binary,
        )

    @staticmethod
    def _pack(
        text, generated, changed, elapsed, safety_fallbacks=0,
        first_step_binary=None,
    ):
        result = {
            "text": text.strip(),
            "token_ids": generated,
            "num_tokens": len(generated),
            "num_changed": changed,
            "change_ratio": changed / max(len(generated), 1),
            "num_safety_fallbacks": int(safety_fallbacks),
            "time_s": elapsed,
            "tokens_per_sec": len(generated) / max(elapsed, 1e-6),
        }
        result.update(first_step_binary or {})
        return result


class BaselinePipeline:
    """Regular greedy baseline — uses model's native generate()."""

    def __init__(self, model_adapter: BaseModelAdapter,
                 do_sample: bool = False, temperature: float = 1.0,
                 top_p: float = 1.0, top_k: Optional[int] = None):
        self.adapter = model_adapter
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

    def generate(self, image: Image.Image, prompt: str,
                 max_new_tokens: int = 512, sample_id: str = "") -> Dict:
        t0 = time.time()
        text = self.adapter.generate_configured(
            image, prompt, max_new_tokens=max_new_tokens,
            do_sample=self.do_sample, temperature=self.temperature,
            top_p=self.top_p, top_k=self.top_k,
        )
        elapsed = time.time() - t0
        return {
            "text": text,
            "token_ids": [],
            "num_tokens": len(self.adapter.tokenizer.encode(text)),
            "num_changed": 0,
            "change_ratio": 0.0,
            "time_s": elapsed,
        }
