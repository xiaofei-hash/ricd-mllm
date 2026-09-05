"""
Base Model Adapter for R-CCR three-path inference.

Adapters implement two interfaces:
  Legacy  : get_three_path_logits()  — no cache, full re-encode every step
  Cached  : prefill() + decode_step_cached()  — KV cache, ~10-50× faster
"""

import torch
from abc import ABC, abstractmethod
from typing import Tuple, List, Any
from dataclasses import dataclass, field
from PIL import Image


@dataclass
class CacheState:
    """KV caches for the three inference paths."""
    deep:    Any = None
    shallow: Any = None
    blind:   Any = None
    # Optional metadata for LLaVA's all-path batched decode.  The blind cache
    # is right-padded to the visual-path prefix length; its padding is masked
    # and its logical RoPE positions continue from ``blind_prefix_length``.
    blind_prefix_length: int = 0
    blind_padding: int = 0
    generated_length: int = 0
    all_paths_batched: bool = False


class BaseModelAdapter(ABC):
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def load_model(self): ...

    # ── legacy (no cache) ──
    @abstractmethod
    def get_three_path_logits(
        self, image: Image.Image, prompt: str, generated_ids: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def generate_greedy(self, image: Image.Image, prompt: str,
                        max_new_tokens: int = 512) -> str: ...

    def generate_configured(self, image: Image.Image, prompt: str,
                            max_new_tokens: int = 512,
                            do_sample: bool = False,
                            temperature: float = 1.0,
                            top_p: float = 1.0,
                            top_k=None) -> str:
        if do_sample:
            raise NotImplementedError(
                "%s does not implement configured sampling" % type(self).__name__
            )
        return self.generate_greedy(image, prompt, max_new_tokens)

    # ── cached ──
    def supports_kv_cache(self) -> bool:
        return False

    def prefill(self, image: Image.Image, prompt: str
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, CacheState]:
        raise NotImplementedError

    def decode_step_cached(self, token_id: int, cs: CacheState
                           ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, CacheState]:
        raise NotImplementedError

    # ── utils ──
    def get_eos_token_ids(self) -> List[int]:
        """Return every token that the model treats as end-of-generation."""
        value = getattr(
            getattr(self.model, "generation_config", None),
            "eos_token_id",
            None,
        )
        if value is None:
            value = getattr(self.tokenizer, "eos_token_id", None)
        if value is None:
            return [2]
        if isinstance(value, int):
            return [value]
        return list(dict.fromkeys(int(token_id) for token_id in value))

    def get_eos_token_id(self) -> int:
        """Backward-compatible primary EOS accessor."""
        return self.get_eos_token_ids()[0]

    def get_vocab_size(self) -> int:
        return self.model.config.vocab_size
