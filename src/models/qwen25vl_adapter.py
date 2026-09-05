"""
Qwen2.5-VL-7B Adapter — KV Cache + Batched Decode

Speed-up:
  prefill  — 3 full forward passes (once)
  decode   — deep+shallow batched via LM backbone, blind separate
             → 2 forward calls per token

Depends: pip install transformers>=4.40 qwen-vl-utils
"""

import torch
import torch.nn.functional as F
from typing import Tuple, List
from PIL import Image

from .base_adapter import BaseModelAdapter, CacheState
from .kv_cache_utils import stack_kv_caches, split_kv_caches


class Qwen25VLAdapter(BaseModelAdapter):

    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 device: str = "cuda", allow_shallow_fallback: bool = False,
                 max_pixels: int = None, response_prefix: str = None,
                 min_new_tokens: int = 0,
                 shallow_layer_ratio: float = None,
                 shallow_layer_index: int = None,
                 attn_implementation: str = None):
        super().__init__("qwen2.5-vl", device)
        if shallow_layer_ratio is not None and shallow_layer_index is not None:
            raise ValueError("Specify only one of shallow_layer_ratio/index")
        self.model_path = model_path
        self.processor = None
        self.allow_shallow_fallback = allow_shallow_fallback
        self.max_pixels = max_pixels
        self.response_prefix = (response_prefix or "").strip()
        self.min_new_tokens = max(int(min_new_tokens), 0)
        self.shallow_layer_ratio = shallow_layer_ratio
        self.shallow_layer_index = shallow_layer_index
        self.attn_implementation = attn_implementation

    def _resolve_shallow_layer(self, total_layers: int) -> int:
        """Resolve a model-relative visual layer, preserving layer 4 by default."""
        if total_layers <= 0:
            raise ValueError("visual encoder has no layers")
        if self.shallow_layer_index is not None:
            index = int(self.shallow_layer_index)
        elif self.shallow_layer_ratio is not None:
            ratio = float(self.shallow_layer_ratio)
            if not 0.0 <= ratio <= 1.0:
                raise ValueError("shallow_layer_ratio must be in [0, 1]")
            index = round(ratio * (total_layers - 1))
        else:
            index = 4
        return max(0, min(index, total_layers - 1))

    def _handle_shallow_failure(self, logits_deep, deep_cache, error):
        """Fail closed for paper runs; optional fallback is debug-only."""
        if not self.allow_shallow_fallback:
            raise RuntimeError(
                "Qwen shallow visual branch failed. Refusing to substitute the "
                "deep branch because that makes V(w) degenerate and invalidates "
                "RiCD results."
            ) from error
        print(f"[Qwen][DEBUG ONLY] shallow branch fallback: {error}")
        return logits_deep.clone(), deep_cache

    # ================================================================ load
    def load_model(self):
        from transformers import AutoProcessor
        print(f"[Qwen2.5-VL] Loading {self.model_path}...")

        model_cls = None
        for name in ["Qwen2_5_VLForConditionalGeneration",
                      "Qwen2VLForConditionalGeneration"]:
            try:
                import transformers
                model_cls = getattr(transformers, name)
                print(f"  Using {name}")
                break
            except AttributeError:
                continue
        if model_cls is None:
            from transformers import AutoModelForCausalLM
            model_cls = AutoModelForCausalLM

        processor_kwargs = {}
        if self.max_pixels is not None:
            processor_kwargs["max_pixels"] = int(self.max_pixels)
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, **processor_kwargs
        )
        if self.max_pixels is not None:
            actual_max = int(self.processor.image_processor.max_pixels)
            if actual_max != int(self.max_pixels):
                raise RuntimeError(
                    "Qwen processor ignored max_pixels=%d (actual=%d)" %
                    (self.max_pixels, actual_max)
                )
            print("  Qwen image max_pixels=%d" % actual_max)
        self.tokenizer = self.processor.tokenizer
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": self.device,
            "low_cpu_mem_usage": True,
        }
        split_attention = self.attn_implementation == "eager_text_sdpa_vision"
        if self.attn_implementation and not split_attention:
            model_kwargs["attn_implementation"] = self.attn_implementation
        elif split_attention:
            model_kwargs["attn_implementation"] = "eager"
        self.model = model_cls.from_pretrained(self.model_path, **model_kwargs)
        if split_attention:
            self._replace_visual_attention_with_sdpa()
        self.model.eval()
        print(f"[Qwen2.5-VL] Loaded. vocab={self.model.config.vocab_size}")

    def _replace_visual_attention_with_sdpa(self):
        """Keep text attention eager for SID while making vision memory-safe."""
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VLVisionSdpaAttention,
        )
        for block in self.model.visual.blocks:
            old = block.attn
            replacement = Qwen2_5_VLVisionSdpaAttention(
                old.num_heads * old.head_dim, num_heads=old.num_heads
            ).to(device=old.qkv.weight.device, dtype=old.qkv.weight.dtype)
            replacement.load_state_dict(old.state_dict())
            block.attn = replacement
        print("  Qwen attention: eager text + SDPA vision")

    # ================================================================ helpers
    def _deep_inputs(self, image, prompt, generated_ids=None):
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": prompt}]}]
        text = self.processor.apply_chat_template(msgs, tokenize=False,
                                                   add_generation_prompt=True)
        text += self.response_prefix
        inp = self.processor(text=[text], images=[image], padding=True,
                             return_tensors="pt")
        inp = {k: v.to(self.device) for k, v in inp.items()}
        if generated_ids:
            g = torch.tensor([generated_ids], dtype=torch.long, device=self.device)
            inp["input_ids"] = torch.cat([inp["input_ids"], g], dim=1)
            if "attention_mask" in inp:
                inp["attention_mask"] = torch.cat(
                    [inp["attention_mask"], torch.ones_like(g)], dim=1)
        return inp

    def _blind_inputs(self, prompt, generated_ids=None):
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(msgs, tokenize=False,
                                                   add_generation_prompt=True)
        text += self.response_prefix
        inp = self.tokenizer(text, return_tensors="pt")
        inp = {k: v.to(self.device) for k, v in inp.items()}
        if generated_ids:
            g = torch.tensor([generated_ids], dtype=torch.long, device=self.device)
            inp["input_ids"] = torch.cat([inp["input_ids"], g], dim=1)
            if "attention_mask" in inp:
                inp["attention_mask"] = torch.cat(
                    [inp["attention_mask"], torch.ones_like(g)], dim=1)
        return inp

    def _shallow_vis(self, pixel_values, grid_thw, layer_index=None):
        """Select an early ViT layer and pass it through the visual merger."""
        visual = self.model.visual
        h = {}
        def hook(m, i, o): h["f"] = o
        layer = (self._resolve_shallow_layer(len(visual.blocks))
                 if layer_index is None else
                 max(0, min(int(layer_index), len(visual.blocks) - 1)))
        handle = visual.blocks[layer].register_forward_hook(hook)
        with torch.no_grad():
            _ = visual(pixel_values, grid_thw=grid_thw)
        handle.remove()
        feats = h["f"]
        if hasattr(visual, "merger"):
            feats = visual.merger(feats)
        return feats

    def _lm_forward(self, input_ids=None, inputs_embeds=None,
                    attention_mask=None, past_key_values=None, use_cache=False):
        """Call Qwen LM backbone + lm_head."""
        out = self.model.model(
            input_ids=input_ids, inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values, use_cache=use_cache)
        logits = self.model.lm_head(out.last_hidden_state)
        return logits, out.past_key_values

    # ================================================================ cached
    def supports_kv_cache(self) -> bool:
        # Qwen2.5-VL uses multimodal 3-D RoPE positions.  The legacy manual
        # cache recurrence below does not preserve the per-path RoPE deltas;
        # after one token it can incorrectly select EOS.  Keep RiCD on the
        # correctness-first full-recompute path until that state is modeled.
        return False

    def prefill(self, image, prompt):
        deep_inp = self._deep_inputs(image, prompt)
        blind_inp = self._blind_inputs(prompt)

        with torch.no_grad():
            # ── Deep ──
            d_out = self.model(**deep_inp, use_cache=True)
            logits_deep = d_out.logits[0, -1, :]
            deep_cache = d_out.past_key_values

            # ── Shallow ──
            pv = deep_inp.get("pixel_values")
            grid = deep_inp.get("image_grid_thw")
            if pv is not None:
                try:
                    sh_feats = self._shallow_vis(pv, grid)
                    ids = deep_inp["input_ids"]
                    emb = self.model.model.embed_tokens(ids)

                    img_tok_id = getattr(self.model.config, "image_token_id",
                                         getattr(getattr(self.model.config, "visual", None),
                                                 "image_token_id", 151655))
                    mask = (ids[0] == img_tok_id)
                    if mask.any():
                        pos = torch.where(mask)[0]
                        n = min(len(pos), sh_feats.shape[0])
                        for i in range(n):
                            emb[0, pos[i]] = sh_feats[i]

                    sl, s_cache = self._lm_forward(
                        inputs_embeds=emb,
                        attention_mask=deep_inp.get("attention_mask"),
                        use_cache=True)
                    logits_shallow = sl[0, -1, :]
                    shallow_cache = s_cache
                except Exception as e:
                    logits_shallow, shallow_cache = self._handle_shallow_failure(
                        logits_deep, deep_cache, e
                    )
            else:
                logits_shallow, shallow_cache = self._handle_shallow_failure(
                    logits_deep, deep_cache,
                    ValueError("processor returned no pixel_values"),
                )

            # ── Blind ──
            bl, b_cache = self._lm_forward(
                input_ids=blind_inp["input_ids"],
                attention_mask=blind_inp.get("attention_mask"),
                use_cache=True)
            logits_blind = bl[0, -1, :]

        cs = CacheState(deep_cache, shallow_cache, b_cache)
        return logits_deep, logits_shallow, logits_blind, cs

    def decode_step_cached(self, token_id, cs):
        tok = torch.tensor([[token_id]], device=self.device)
        with torch.no_grad():
            try:
                tok2 = tok.expand(2, -1)
                merged = stack_kv_caches(cs.deep, cs.shallow)
                logits, kv = self._lm_forward(input_ids=tok2,
                                               past_key_values=merged,
                                               use_cache=True)
                ld, ls = logits[0,-1,:], logits[1,-1,:]
                cs.deep, cs.shallow = split_kv_caches(kv)
            except Exception:
                ld_all, d_kv = self._lm_forward(input_ids=tok,
                                                 past_key_values=cs.deep,
                                                 use_cache=True)
                ls_all, s_kv = self._lm_forward(input_ids=tok,
                                                 past_key_values=cs.shallow,
                                                 use_cache=True)
                ld, ls = ld_all[0,-1,:], ls_all[0,-1,:]
                cs.deep, cs.shallow = d_kv, s_kv

            lb_all, b_kv = self._lm_forward(input_ids=tok,
                                             past_key_values=cs.blind,
                                             use_cache=True)
            lb = lb_all[0,-1,:]
            cs.blind = b_kv

        return ld, ls, lb, cs

    # ================================================================ legacy
    def get_layercd_logits(self, image, prompt, generated_ids=None,
                           vision_layer=1):
        """Deep/shallow vision-encoder paths for LayerCD."""
        deep_inputs = self._deep_inputs(image, prompt, generated_ids)
        with torch.no_grad():
            deep = self.model(**deep_inputs).logits[0, -1, :]
            pixel_values = deep_inputs.get("pixel_values")
            grid = deep_inputs.get("image_grid_thw")
            if pixel_values is None:
                raise RuntimeError("Qwen LayerCD requires pixel_values")
            shallow_features = self._shallow_vis(
                pixel_values, grid, layer_index=vision_layer
            )
            ids = deep_inputs["input_ids"]
            embeddings = self.model.model.embed_tokens(ids)
            image_token = getattr(self.model.config, "image_token_id", 151655)
            positions = torch.where(ids[0] == image_token)[0]
            if not len(positions):
                raise RuntimeError("Qwen LayerCD found no image token positions")
            if len(positions) != shallow_features.shape[0]:
                raise RuntimeError(
                    "Qwen LayerCD image token/feature mismatch: %d vs %d" %
                    (len(positions), shallow_features.shape[0])
                )
            embeddings = embeddings.clone()
            embeddings[0, positions] = shallow_features.to(embeddings.dtype)
            shallow_logits, _ = self._lm_forward(
                inputs_embeds=embeddings,
                attention_mask=deep_inputs.get("attention_mask"),
            )
            shallow = shallow_logits[0, -1, :]
        return deep, shallow

    def get_three_path_logits(self, image, prompt, generated_ids=None):
        deep_inp = self._deep_inputs(image, prompt, generated_ids)
        blind_inp = self._blind_inputs(prompt, generated_ids)

        with torch.no_grad():
            ld = self.model(**deep_inp).logits[0,-1,:]

            bl, _ = self._lm_forward(input_ids=blind_inp["input_ids"],
                                      attention_mask=blind_inp.get("attention_mask"))
            lb = bl[0,-1,:]

            pv = deep_inp.get("pixel_values")
            grid = deep_inp.get("image_grid_thw")
            if pv is not None:
                try:
                    sh = self._shallow_vis(pv, grid)
                    ids = deep_inp["input_ids"]
                    emb = self.model.model.embed_tokens(ids)
                    img_tok = getattr(self.model.config, "image_token_id", 151655)
                    mask = (ids[0] == img_tok)
                    if mask.any():
                        pos = torch.where(mask)[0]
                        n = min(len(pos), sh.shape[0])
                        for i in range(n):
                            emb[0, pos[i]] = sh[i]
                    sl, _ = self._lm_forward(
                        inputs_embeds=emb,
                        attention_mask=deep_inp.get("attention_mask"))
                    ls = sl[0,-1,:]
                except Exception as e:
                    ls, _ = self._handle_shallow_failure(ld, None, e)
            else:
                ls, _ = self._handle_shallow_failure(
                    ld, None, ValueError("processor returned no pixel_values")
                )

        return ld, ls, lb

    def generate_greedy(self, image, prompt, max_new_tokens=512):
        return self.generate_configured(
            image, prompt, max_new_tokens=max_new_tokens, do_sample=False,
        )

    def generate_configured(self, image, prompt, max_new_tokens=512,
                            do_sample=False, temperature=1.0,
                            top_p=1.0, top_k=None):
        """Run the native Qwen-VL generator with the registered decode policy."""
        inp = self._deep_inputs(image, prompt)
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min(self.min_new_tokens, max_new_tokens),
            "do_sample": do_sample,
            "use_cache": True,
        }
        if do_sample:
            generation_kwargs.update({
                "temperature": temperature,
                "top_p": top_p,
            })
            if top_k is not None:
                generation_kwargs["top_k"] = top_k
        with torch.no_grad():
            out = self.model.generate(**inp, **generation_kwargs)
        n = inp["input_ids"].shape[1]
        continuation = self.tokenizer.decode(
            out[0, n:], skip_special_tokens=True
        ).strip()
        if self.response_prefix:
            return (self.response_prefix + " " + continuation).strip()
        return continuation
