"""
LLaVA-1.5-7B Adapter — KV Cache + Batched Decode

Speed-up strategy:
  1. optional experimental prefill batch —
       • extract deep+shallow CLIP features in one vision-tower pass
       • deep+shallow language prefill in one batch=2 forward
       • blind prefill remains separate because its context is shorter
  2. decode_step_cached() —
       • deep+shallow batched into 1 forward (batch=2) with stacked KV cache
       • blind: 1 separate forward (different seq length)
       → 2 forward calls per token instead of 3 independent calls

Depends on: pip install llava  (or git clone + pip install -e .)
"""

import os
import warnings

import torch
import torch.nn.functional as F
from typing import Tuple, List
from PIL import Image

from .base_adapter import BaseModelAdapter, CacheState
from .kv_cache_utils import (
    cache_seq_length,
    pad_kv_cache_right,
    split_kv_caches,
    split_kv_caches_many,
    stack_kv_caches,
    stack_kv_caches_many,
)


class LLaVAAdapter(BaseModelAdapter):

    def __init__(self, model_path: str = "liuhaotian/llava-v1.5-7b",
                 device: str = "cuda"):
        super().__init__("llava-1.5-7b", device)
        self.model_path = model_path
        self.image_processor = None
        self.conv_mode = "llava_v1"

    # ================================================================ load
    def load_model(self):
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path

        model_name = get_model_name_from_path(self.model_path)
        print(f"[LLaVA] Loading {self.model_path}  (name={model_name}) ...")

        self.tokenizer, self.model, self.image_processor, self.context_len = \
            load_pretrained_model(self.model_path, model_base=None,
                                  model_name=model_name)
        self.model.eval()

        mn = model_name.lower()
        if "llama-2" in mn:      self.conv_mode = "llava_llama_2"
        elif "mistral" in mn:    self.conv_mode = "mistral_instruct"
        elif "v1.6-34b" in mn:   self.conv_mode = "chatml_direct"
        elif "v1" in mn:         self.conv_mode = "llava_v1"
        else:                    self.conv_mode = "llava_v0"

        print(f"[LLaVA] Loaded.  conv_mode={self.conv_mode}")

    # ================================================================ helpers
    def _build_prompt(self, question: str, with_image: bool) -> str:
        from llava.conversation import conv_templates
        from llava.constants import (DEFAULT_IMAGE_TOKEN,
                                     DEFAULT_IM_START_TOKEN,
                                     DEFAULT_IM_END_TOKEN)
        conv = conv_templates[self.conv_mode].copy()
        if with_image:
            if getattr(self.model.config, "mm_use_im_start_end", False):
                img = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            else:
                img = DEFAULT_IMAGE_TOKEN
            conv.append_message(conv.roles[0], img + "\n" + question)
        else:
            conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def _tokenize(self, prompt: str, has_image: bool) -> torch.Tensor:
        if has_image:
            from llava.mm_utils import tokenizer_image_token
            from llava.constants import IMAGE_TOKEN_INDEX
            ids = tokenizer_image_token(prompt, self.tokenizer,
                                        IMAGE_TOKEN_INDEX, return_tensors="pt")
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
        else:
            ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        return ids.to(self.device)

    def _process_image(self, image: Image.Image) -> torch.Tensor:
        from llava.mm_utils import process_images
        t = process_images([image], self.image_processor, self.model.config)
        return t.to(dtype=torch.float16, device=self.device)

    def _shallow_features(self, image_tensor: torch.Tensor,
                          layer_index: int = 4) -> torch.Tensor:
        """Project CLIP ViT features from an explicitly selected layer."""
        vt = self.model.get_vision_tower()
        with torch.no_grad():
            out = vt.vision_tower(image_tensor, output_hidden_states=True)
            index = max(0, min(int(layer_index), len(out.hidden_states) - 1))
            feats = out.hidden_states[index]
        sel = getattr(self.model.config, "mm_vision_select_feature", "patch")
        if sel == "patch":
            feats = feats[:, 1:]
        return self.model.get_model().mm_projector(feats)

    def _deep_shallow_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Return exact deep/shallow visual features from one CLIP pass.

        The first batch element follows LLaVA's configured vision-selection
        layer (deep path); the second uses RiCD's layer-4 early exit.  Both are
        projected together so ``prepare_inputs_labels_for_multimodal`` can
        consume them as an ordinary batch of two images.
        """
        vt = self.model.get_vision_tower()
        pixel_values = image_tensor[:1].to(device=vt.device, dtype=vt.dtype)
        with torch.no_grad():
            out = vt.vision_tower(pixel_values, output_hidden_states=True)
            deep = vt.feature_select(out).to(image_tensor.dtype)
            shallow = out.hidden_states[4]
            if vt.select_feature == "patch":
                shallow = shallow[:, 1:]
            elif vt.select_feature != "cls_patch":
                raise ValueError(
                    "Unexpected vision feature selection: %s" % vt.select_feature
                )
            shallow = shallow.to(image_tensor.dtype)
            features = torch.cat([deep, shallow], dim=0)
            return self.model.get_model().mm_projector(features)

    # ================================================================ cached
    def supports_kv_cache(self) -> bool:
        return True

    def _prepare_prefill_inputs(self, image: Image.Image, prompt: str):
        p_img = self._build_prompt(prompt, with_image=True)
        p_blind = self._build_prompt(prompt, with_image=False)
        return (
            self._tokenize(p_img, has_image=True),
            self._tokenize(p_blind, has_image=False),
            self._process_image(image),
        )

    def prefill_sequential(self, image: Image.Image, prompt: str):
        """Reference three-forward prefill retained for equivalence tests."""
        ids_img, ids_blind, img_t = self._prepare_prefill_inputs(image, prompt)

        with torch.no_grad():
            d_out = self.model(input_ids=ids_img, images=img_t,
                               use_cache=True)
            logits_deep = d_out.logits[0, -1, :]

            _orig = self.model.encode_images
            self.model.encode_images = lambda imgs: self._shallow_features(imgs)
            try:
                s_out = self.model(input_ids=ids_img, images=img_t,
                                   use_cache=True)
            finally:
                self.model.encode_images = _orig
            logits_shallow = s_out.logits[0, -1, :]

            b_out = self.model(input_ids=ids_blind, images=None,
                               use_cache=True)
            logits_blind = b_out.logits[0, -1, :]

        return logits_deep, logits_shallow, logits_blind, CacheState(
            deep=d_out.past_key_values,
            shallow=s_out.past_key_values,
            blind=b_out.past_key_values,
        )

    def prefill_batched(self, image: Image.Image, prompt: str):
        """Batch the two vision-conditioned paths during prefill."""
        ids_img, ids_blind, img_t = self._prepare_prefill_inputs(image, prompt)
        ids_vision = ids_img.expand(2, -1)
        images_vision = img_t.expand(2, -1, -1, -1)

        with torch.no_grad():
            original_encode_images = self.model.encode_images
            self.model.encode_images = (
                lambda imgs: self._deep_shallow_features(imgs[:1])
            )
            try:
                ds_out = self.model(
                    input_ids=ids_vision,
                    images=images_vision,
                    use_cache=True,
                )
            finally:
                self.model.encode_images = original_encode_images

            b_out = self.model(
                input_ids=ids_blind,
                images=None,
                use_cache=True,
            )

        deep_cache, shallow_cache = split_kv_caches(ds_out.past_key_values)
        return (
            ds_out.logits[0, -1, :],
            ds_out.logits[1, -1, :],
            b_out.logits[0, -1, :],
            CacheState(
                deep=deep_cache,
                shallow=shallow_cache,
                blind=b_out.past_key_values,
            ),
        )

    def prefill(self, image: Image.Image, prompt: str):
        """
        Full prompt through all three paths, batching deep+shallow by default.

        Set ``RICD_ENABLE_BATCH_PREFILL=1`` to test batch=2 prefill.  It is
        disabled by default because it did not improve RTX 4090 latency.
        """
        # Batch=2 prefill is retained as an opt-in experiment.  On RTX 4090 it
        # increases peak memory without improving latency, so the measured
        # sequential reference remains the production default.
        if os.environ.get("RICD_ENABLE_BATCH_PREFILL") != "1":
            return self.prefill_sequential(image, prompt)
        try:
            return self.prefill_batched(image, prompt)
        except Exception as exc:
            warnings.warn(
                "Batched RiCD prefill failed; using sequential reference: %r" % exc,
                RuntimeWarning,
            )
            return self.prefill_sequential(image, prompt)

    def _decode_step_all_paths_batched(self, token_id: int, cs: CacheState):
        """Decode deep, shallow and blind in one padded batch=3 call.

        The blind prefix is shorter because it has no visual tokens.  Its KV
        tensors are right-padded once, while ``attention_mask`` hides those
        slots and explicit ``position_ids`` preserve its logical RoPE index.
        """
        deep_length = cache_seq_length(cs.deep)
        shallow_length = cache_seq_length(cs.shallow)
        if deep_length != shallow_length:
            raise ValueError(
                "Deep/shallow cache lengths differ: %d vs %d" %
                (deep_length, shallow_length)
            )

        if not cs.all_paths_batched:
            blind_length = cache_seq_length(cs.blind)
            padding = deep_length - blind_length
            if padding < 0:
                raise ValueError("Blind cache is longer than visual caches")
            cs.blind_prefix_length = blind_length
            cs.blind_padding = padding
            cs.generated_length = 0
            cs.blind = pad_kv_cache_right(cs.blind, padding)
            cs.all_paths_batched = True

        blind_physical_length = cache_seq_length(cs.blind)
        if blind_physical_length != deep_length:
            raise ValueError(
                "Padded blind cache length %d != visual cache length %d" %
                (blind_physical_length, deep_length)
            )

        merged = stack_kv_caches_many([cs.deep, cs.shallow, cs.blind])
        tokens = torch.tensor([[token_id]], device=self.device).expand(3, -1)

        mask = torch.ones(
            (3, deep_length + 1), dtype=torch.long, device=self.device
        )
        if cs.blind_padding:
            pad_start = cs.blind_prefix_length
            pad_end = pad_start + cs.blind_padding
            mask[2, pad_start:pad_end] = 0

        position_ids = torch.tensor(
            [
                [deep_length],
                [deep_length],
                [cs.blind_prefix_length + cs.generated_length],
            ],
            dtype=torch.long,
            device=self.device,
        )

        output = self.model(
            input_ids=tokens,
            images=None,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_values=merged,
            use_cache=True,
        )
        cs.deep, cs.shallow, cs.blind = split_kv_caches_many(
            output.past_key_values, 3
        )
        cs.generated_length += 1
        return (
            output.logits[0, -1, :],
            output.logits[1, -1, :],
            output.logits[2, -1, :],
            cs,
        )

    def decode_step_cached(self, token_id: int, cs: CacheState):
        """
        Single-token decode with KV cache.
        Batches deep+shallow (same seq length) into one forward call.
        """
        if os.environ.get("RICD_DISABLE_BATCH_ALL_DECODE") != "1":
            return self._decode_step_all_paths_batched(token_id, cs)

        tok = torch.tensor([[token_id]], device=self.device)

        with torch.no_grad():
            try:
                # ── deep + shallow batched ──
                tok2 = tok.expand(2, -1)                           # [2, 1]
                merged = stack_kv_caches(cs.deep, cs.shallow)
                ds_out = self.model(input_ids=tok2, images=None,
                                    past_key_values=merged, use_cache=True)
                logits_deep    = ds_out.logits[0, -1, :]
                logits_shallow = ds_out.logits[1, -1, :]
                cs.deep, cs.shallow = split_kv_caches(ds_out.past_key_values)
            except Exception:
                # fallback: sequential
                d_out = self.model(input_ids=tok, images=None,
                                   past_key_values=cs.deep, use_cache=True)
                logits_deep = d_out.logits[0, -1, :]
                cs.deep = d_out.past_key_values

                s_out = self.model(input_ids=tok, images=None,
                                   past_key_values=cs.shallow, use_cache=True)
                logits_shallow = s_out.logits[0, -1, :]
                cs.shallow = s_out.past_key_values

            # ── blind (different seq length, always separate) ──
            b_out = self.model(input_ids=tok, images=None,
                               past_key_values=cs.blind, use_cache=True)
            logits_blind = b_out.logits[0, -1, :]
            cs.blind = b_out.past_key_values

        return logits_deep, logits_shallow, logits_blind, cs

    # ================================================================ legacy (no cache)
    def get_layercd_logits(self, image, prompt, generated_ids=None,
                           vision_layer=1):
        """Deep/shallow visual logits for Layer Contrastive Decoding."""
        p_img = self._build_prompt(prompt, with_image=True)
        ids_img = self._tokenize(p_img, has_image=True)
        if generated_ids:
            g = torch.tensor([generated_ids], dtype=torch.long,
                             device=self.device)
            ids_img = torch.cat([ids_img, g], dim=1)
        img_t = self._process_image(image)
        with torch.no_grad():
            deep = self.model(input_ids=ids_img, images=img_t).logits[0, -1, :]
            original_encode_images = self.model.encode_images
            self.model.encode_images = lambda imgs: self._shallow_features(
                imgs, layer_index=vision_layer
            )
            try:
                shallow = self.model(
                    input_ids=ids_img, images=img_t
                ).logits[0, -1, :]
            finally:
                self.model.encode_images = original_encode_images
        return deep, shallow

    def get_three_path_logits(self, image, prompt, generated_ids=None):
        p_img   = self._build_prompt(prompt, with_image=True)
        p_blind = self._build_prompt(prompt, with_image=False)
        ids_img   = self._tokenize(p_img, has_image=True)
        ids_blind = self._tokenize(p_blind, has_image=False)

        if generated_ids:
            g = torch.tensor([generated_ids], dtype=torch.long, device=self.device)
            ids_img   = torch.cat([ids_img, g], dim=1)
            ids_blind = torch.cat([ids_blind, g], dim=1)

        img_t = self._process_image(image)
        sizes = [image.size]

        with torch.no_grad():
            logits_deep = self.model(
                input_ids=ids_img, images=img_t
            ).logits[0, -1, :]

            logits_blind = self.model(
                input_ids=ids_blind, images=None
            ).logits[0, -1, :]

            _orig = self.model.encode_images
            self.model.encode_images = lambda imgs: self._shallow_features(imgs)
            try:
                logits_shallow = self.model(
                    input_ids=ids_img, images=img_t
                ).logits[0, -1, :]
            finally:
                self.model.encode_images = _orig

        return logits_deep, logits_shallow, logits_blind

    # ================================================================ generate
    def generate_greedy(self, image, prompt, max_new_tokens=512):
        return self.generate_configured(
            image, prompt, max_new_tokens=max_new_tokens, do_sample=False,
        )

    def generate_configured(self, image, prompt, max_new_tokens=512,
                            do_sample=False, temperature=1.0,
                            top_p=1.0, top_k=None):
        p = self._build_prompt(prompt, with_image=True)
        ids = self._tokenize(p, has_image=True)
        img_t = self._process_image(image)

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "use_cache": True,
        }
        if do_sample:
            generation_kwargs.update({
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            })
        with torch.no_grad():
            out = self.model.generate(
                ids, images=img_t, **generation_kwargs,
            )
        return self.tokenizer.decode(out[0, ids.shape[1]:],
                                     skip_special_tokens=True).strip()
