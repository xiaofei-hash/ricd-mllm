"""
InstructBLIP Adapter — KV Cache + Batched Decode

HF backend  : InstructBlipForConditionalGeneration
LAVIS backend: blip2_vicuna_instruct (fallback)

Speed-up:
  prefill  — build visual + text embeds once, feed to LM with use_cache=True
  decode   — single-token LM calls with past_key_values
             deep+shallow batched (same cache length), blind separate
"""

import torch
import torch.nn.functional as F
from typing import Tuple, List
from PIL import Image

from .base_adapter import BaseModelAdapter, CacheState
from .kv_cache_utils import stack_kv_caches, split_kv_caches


class InstructBLIPAdapter(BaseModelAdapter):

    def __init__(self, model_path: str = "Salesforce/instructblip-vicuna-7b",
                 device: str = "cuda", shallow_layer_ratio: float = None,
                 shallow_layer_index: int = None,
                 attn_implementation: str = None):
        super().__init__("instructblip", device)
        if shallow_layer_ratio is not None and shallow_layer_index is not None:
            raise ValueError("Specify only one of shallow_layer_ratio/index")
        self.model_path = model_path
        self.shallow_layer_ratio = shallow_layer_ratio
        self.shallow_layer_index = shallow_layer_index
        self.attn_implementation = attn_implementation
        self._backend = None
        self.vis_processor = None
        self._lm = None   # language model reference (set after load)

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

    # ================================================================ load
    def load_model(self):
        try:
            self._load_hf()
        except Exception as e:
            print(f"[InstructBLIP] HF failed ({e}), trying LAVIS...")
            self._load_lavis()

    def _load_hf(self):
        from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor
        print(f"[InstructBLIP] Loading via HF: {self.model_path}...")
        # The shared cache may contain a tokenizer.json produced by a newer
        # tokenizers release than the LLaVA-compatible environment provides.
        # The slow SentencePiece tokenizer is format-stable and produces the
        # same token IDs for this checkpoint.
        self.processor = InstructBlipProcessor.from_pretrained(
            self.model_path, use_fast=False,
        )
        self.tokenizer = self.processor.tokenizer
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": self.device,
            "low_cpu_mem_usage": True,
        }
        if self.attn_implementation:
            model_kwargs["attn_implementation"] = self.attn_implementation
        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            self.model_path, **model_kwargs
        )
        self.model.eval()
        self._lm = self.model.language_model
        self._backend = "hf"
        print("[InstructBLIP] Loaded (HF).")

    def _load_lavis(self):
        from lavis.models import load_model_and_preprocess
        print("[InstructBLIP] Loading via LAVIS...")
        self.model, vis_procs, _ = load_model_and_preprocess(
            name="blip2_vicuna_instruct", model_type="vicuna7b",
            is_eval=True, device=self.device,
        )
        self.vis_processor = vis_procs["eval"]
        self.tokenizer = self.model.llm_tokenizer
        self._lm = self.model.llm_model
        self._backend = "lavis"
        print("[InstructBLIP] Loaded (LAVIS).")

    # ================================================================ embed builders
    def _build_embeds_hf(self, image, prompt, generated_ids=None):
        """Build (deep_embeds, deep_attn), (shallow_embeds, shallow_attn), (blind_embeds, blind_attn)."""
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        vision_dtype = next(self.model.vision_model.parameters()).dtype
        pv = inputs["pixel_values"].to(dtype=vision_dtype)
        qf_ids  = inputs.get("qformer_input_ids")
        qf_mask = inputs.get("qformer_attention_mask")
        input_ids = inputs["input_ids"]
        attn = inputs.get("attention_mask")

        if generated_ids:
            g = torch.tensor([generated_ids], dtype=torch.long, device=self.device)
            input_ids = torch.cat([input_ids, g], dim=1)
            if attn is not None:
                attn = torch.cat([attn, torch.ones_like(g)], dim=1)

        text_emb = self._lm.get_input_embeddings()(input_ids)

        # Build the genuinely image-free branch from the tokenizer alone.
        # Newer InstructBLIP processors insert image placeholder IDs into the
        # language input.  Reusing those IDs for the blind path is not blind.
        blind_tokens = self.tokenizer(prompt, return_tensors="pt")
        blind_ids = blind_tokens.input_ids.to(self.device)
        blind_attn = blind_tokens.attention_mask.to(self.device)
        if generated_ids:
            blind_g = torch.tensor([generated_ids], dtype=torch.long,
                                   device=self.device)
            blind_ids = torch.cat([blind_ids, blind_g], dim=1)
            blind_attn = torch.cat([blind_attn, torch.ones_like(blind_g)], dim=1)
        blind_emb = self._lm.get_input_embeddings()(blind_ids)

        def _qformer_pass(vision_feats):
            qt = self.model.query_tokens.expand(vision_feats.shape[0], -1, -1)
            enc_attn = torch.ones(vision_feats.shape[:-1], dtype=torch.long, device=self.device)
            query_attn = torch.ones(qt.size()[:-1], dtype=torch.long,
                                    device=self.device)
            qformer_kwargs = {
                "query_embeds": qt,
                "encoder_hidden_states": vision_feats,
                "encoder_attention_mask": enc_attn,
            }
            if qf_ids is not None:
                qformer_kwargs["input_ids"] = qf_ids
                qformer_kwargs["attention_mask"] = torch.cat(
                    [query_attn, qf_mask], dim=1,
                )
            else:
                qformer_kwargs["attention_mask"] = query_attn
            qf_out = self.model.qformer(**qformer_kwargs)
            query_hidden = qf_out.last_hidden_state[:, :qt.shape[1], :]
            proj = self.model.language_projection(query_hidden)
            proj_attn = torch.ones(proj.size()[:-1], dtype=torch.long, device=self.device)
            return proj, proj_attn

        def _compose_language_inputs(visual_projection, visual_attention):
            image_token_index = getattr(self.model.config, "image_token_index", None)
            placeholder_count = (
                int((input_ids == image_token_index).sum().item())
                if image_token_index is not None else 0
            )
            # Older InstructBLIP processors do not insert image placeholder
            # IDs.  This checkpoint uses that legacy format, for which the HF
            # model prepends the projected query tokens to the text embeddings.
            if image_token_index is None or placeholder_count == 0:
                return (torch.cat([visual_projection, text_emb], dim=1),
                        torch.cat([visual_attention, attn], dim=1))
            if placeholder_count != visual_projection.shape[1]:
                raise RuntimeError(
                    "InstructBLIP image placeholder count %d does not match "
                    "%d projected query tokens" %
                    (placeholder_count, visual_projection.shape[1])
                )
            composed = text_emb.clone()
            special_mask = (input_ids == image_token_index).unsqueeze(-1).expand_as(composed)
            projected = visual_projection.to(device=composed.device, dtype=composed.dtype)
            composed[special_mask] = projected.flatten()
            return composed, attn

        # Deep vision
        deep_vis = self._hf_full_vision(pv)
        deep_proj, dp_attn = _qformer_pass(deep_vis)
        deep_emb, deep_attn = _compose_language_inputs(deep_proj, dp_attn)

        # Shallow vision
        shallow_vis = self._hf_shallow_vision(pv)
        sh_proj, sp_attn = _qformer_pass(shallow_vis)
        sh_emb, sh_attn = _compose_language_inputs(sh_proj, sp_attn)

        return (deep_emb, deep_attn), (sh_emb, sh_attn), (blind_emb, blind_attn)

    def _hf_full_vision(self, pixel_values):
        vis = self.model.vision_model(pixel_values, return_dict=True)
        # InstructBlipVisionModel.forward already applies post_layernorm.
        # Applying it again here changes the deep branch and makes its logits
        # disagree with model(**processor_inputs).
        return vis.last_hidden_state

    def _hf_shallow_vision(self, pixel_values, layer_index=None):
        h = {}
        def hook(m, i, o): h["f"] = o[0]
        layers = self.model.vision_model.encoder.layers
        layer = (self._resolve_shallow_layer(len(layers)) if layer_index is None
                 else max(0, min(int(layer_index), len(layers) - 1)))
        handle = layers[layer].register_forward_hook(hook)
        with torch.no_grad():
            _ = self.model.vision_model(pixel_values, return_dict=True)
        handle.remove()
        feats = h["f"]
        if hasattr(self.model.vision_model, "post_layernorm"):
            feats = self.model.vision_model.post_layernorm(feats)
        return feats

    def _build_embeds_lavis(self, image, prompt, generated_ids=None):
        img_t = self.vis_processor(image).unsqueeze(0).to(self.device)

        # full + shallow vision features
        deep_feats   = self.model.ln_vision(self.model.visual_encoder(img_t))
        shallow_feats = self._lavis_shallow_vision(img_t)
        shallow_feats = self.model.ln_vision(shallow_feats)

        qt = self.model.query_tokens.expand(1, -1, -1)

        def _qf(feats):
            enc_attn = torch.ones(feats.size()[:-1], dtype=torch.long, device=self.device)
            out = self.model.Qformer.bert(
                query_embeds=qt, encoder_hidden_states=feats,
                encoder_attention_mask=enc_attn, return_dict=True)
            proj = self.model.llm_proj(out.last_hidden_state)
            pa = torch.ones(proj.size()[:-1], dtype=torch.long, device=self.device)
            return proj, pa

        deep_proj, dp_a = _qf(deep_feats)
        sh_proj, sp_a = _qf(shallow_feats)

        toks = self.model.llm_tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        ids, ta = toks.input_ids, toks.attention_mask
        if generated_ids:
            g = torch.tensor([generated_ids], dtype=torch.long, device=self.device)
            ids = torch.cat([ids, g], dim=1)
            ta = torch.cat([ta, torch.ones_like(g)], dim=1)

        text_emb = self._lm.get_input_embeddings()(ids)

        return (
            (torch.cat([deep_proj, text_emb], 1), torch.cat([dp_a, ta], 1)),
            (torch.cat([sh_proj, text_emb], 1),   torch.cat([sp_a, ta], 1)),
            (text_emb, ta),
        )

    def _lavis_shallow_vision(self, img_t, layer_index=None):
        h = {}
        def hook(m, i, o): h["f"] = o
        blocks = self.model.visual_encoder.blocks
        layer = (self._resolve_shallow_layer(len(blocks)) if layer_index is None
                 else max(0, min(int(layer_index), len(blocks) - 1)))
        handle = blocks[layer].register_forward_hook(hook)
        with torch.no_grad():
            _ = self.model.visual_encoder(img_t)
        handle.remove()
        return h["f"]

    # ================================================================ cached
    def supports_kv_cache(self) -> bool:
        return True

    def prefill(self, image, prompt):
        if self._backend == "hf":
            (de, da), (se, sa), (be, ba) = self._build_embeds_hf(image, prompt)
        else:
            (de, da), (se, sa), (be, ba) = self._build_embeds_lavis(image, prompt)

        with torch.no_grad():
            d_out = self._lm(inputs_embeds=de, attention_mask=da, use_cache=True)
            s_out = self._lm(inputs_embeds=se, attention_mask=sa, use_cache=True)
            b_out = self._lm(inputs_embeds=be, attention_mask=ba, use_cache=True)

        cs = CacheState(d_out.past_key_values, s_out.past_key_values,
                        b_out.past_key_values)
        return d_out.logits[0,-1,:], s_out.logits[0,-1,:], b_out.logits[0,-1,:], cs

    def decode_step_cached(self, token_id, cs):
        tok = torch.tensor([[token_id]], device=self.device)
        with torch.no_grad():
            try:
                # batch deep+shallow
                tok2 = tok.expand(2, -1)
                merged = stack_kv_caches(cs.deep, cs.shallow)
                ds = self._lm(input_ids=tok2, past_key_values=merged, use_cache=True)
                ld, ls = ds.logits[0,-1,:], ds.logits[1,-1,:]
                cs.deep, cs.shallow = split_kv_caches(ds.past_key_values)
            except Exception:
                d = self._lm(input_ids=tok, past_key_values=cs.deep, use_cache=True)
                s = self._lm(input_ids=tok, past_key_values=cs.shallow, use_cache=True)
                ld, ls = d.logits[0,-1,:], s.logits[0,-1,:]
                cs.deep, cs.shallow = d.past_key_values, s.past_key_values

            b = self._lm(input_ids=tok, past_key_values=cs.blind, use_cache=True)
            lb = b.logits[0,-1,:]
            cs.blind = b.past_key_values

        return ld, ls, lb, cs

    # ================================================================ legacy
    def get_layercd_logits(self, image, prompt, generated_ids=None,
                           vision_layer=1):
        """Deep/shallow vision-encoder paths for LayerCD."""
        if self._backend == "hf":
            inputs = self.processor(images=image, text=prompt,
                                    return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            vision_dtype = next(self.model.vision_model.parameters()).dtype
            pv = inputs["pixel_values"].to(dtype=vision_dtype)
            qf_ids = inputs.get("qformer_input_ids")
            qf_mask = inputs.get("qformer_attention_mask")
            input_ids = inputs["input_ids"]
            attn = inputs.get("attention_mask")
            if generated_ids:
                generated = torch.tensor(
                    [generated_ids], dtype=torch.long, device=self.device
                )
                input_ids = torch.cat([input_ids, generated], dim=1)
                attn = torch.cat([attn, torch.ones_like(generated)], dim=1)
            text_emb = self._lm.get_input_embeddings()(input_ids)

            def project(vision_feats):
                query = self.model.query_tokens.expand(
                    vision_feats.shape[0], -1, -1
                )
                query_attn = torch.ones(
                    query.size()[:-1], dtype=torch.long, device=self.device
                )
                kwargs = {
                    "query_embeds": query,
                    "encoder_hidden_states": vision_feats,
                    "encoder_attention_mask": torch.ones(
                        vision_feats.shape[:-1], dtype=torch.long,
                        device=self.device,
                    ),
                }
                if qf_ids is not None:
                    kwargs.update({
                        "input_ids": qf_ids,
                        "attention_mask": torch.cat(
                            [query_attn, qf_mask], dim=1
                        ),
                    })
                else:
                    kwargs["attention_mask"] = query_attn
                qformer = self.model.qformer(**kwargs)
                hidden = qformer.last_hidden_state[:, :query.shape[1], :]
                projection = self.model.language_projection(hidden)
                projection_attn = torch.ones(
                    projection.size()[:-1], dtype=torch.long,
                    device=self.device,
                )
                image_token_index = getattr(
                    self.model.config, "image_token_index", None
                )
                placeholder_count = (
                    int((input_ids == image_token_index).sum().item())
                    if image_token_index is not None else 0
                )
                if image_token_index is None or placeholder_count == 0:
                    return (torch.cat([projection, text_emb], dim=1),
                            torch.cat([projection_attn, attn], dim=1))
                composed = text_emb.clone()
                mask = (input_ids == image_token_index).unsqueeze(-1).expand_as(
                    composed
                )
                composed[mask] = projection.to(composed.dtype).flatten()
                return composed, attn

            deep_inputs = project(self._hf_full_vision(pv))
            shallow_inputs = project(
                self._hf_shallow_vision(pv, layer_index=vision_layer)
            )
        else:
            img_t = self.vis_processor(image).unsqueeze(0).to(self.device)
            deep_feats = self.model.ln_vision(self.model.visual_encoder(img_t))
            shallow_feats = self.model.ln_vision(
                self._lavis_shallow_vision(img_t, layer_index=vision_layer)
            )
            query = self.model.query_tokens.expand(1, -1, -1)

            def project(vision_feats):
                qformer = self.model.Qformer.bert(
                    query_embeds=query,
                    encoder_hidden_states=vision_feats,
                    encoder_attention_mask=torch.ones(
                        vision_feats.size()[:-1], dtype=torch.long,
                        device=self.device,
                    ),
                    return_dict=True,
                )
                projection = self.model.llm_proj(qformer.last_hidden_state)
                tokens = self.model.llm_tokenizer(
                    prompt, return_tensors="pt", padding=True
                ).to(self.device)
                ids, attn = tokens.input_ids, tokens.attention_mask
                if generated_ids:
                    generated = torch.tensor(
                        [generated_ids], dtype=torch.long, device=self.device
                    )
                    ids = torch.cat([ids, generated], dim=1)
                    attn = torch.cat([attn, torch.ones_like(generated)], dim=1)
                text_emb = self._lm.get_input_embeddings()(ids)
                projection_attn = torch.ones(
                    projection.size()[:-1], dtype=torch.long,
                    device=self.device,
                )
                return (torch.cat([projection, text_emb], dim=1),
                        torch.cat([projection_attn, attn], dim=1))

            deep_inputs = project(deep_feats)
            shallow_inputs = project(shallow_feats)

        with torch.no_grad():
            deep = self._lm(
                inputs_embeds=deep_inputs[0], attention_mask=deep_inputs[1]
            ).logits[0, -1, :]
            shallow = self._lm(
                inputs_embeds=shallow_inputs[0],
                attention_mask=shallow_inputs[1],
            ).logits[0, -1, :]
        return deep, shallow

    def get_three_path_logits(self, image, prompt, generated_ids=None):
        if self._backend == "hf":
            (de,da),(se,sa),(be,ba) = self._build_embeds_hf(image, prompt, generated_ids)
        else:
            (de,da),(se,sa),(be,ba) = self._build_embeds_lavis(image, prompt, generated_ids)
        with torch.no_grad():
            ld = self._lm(inputs_embeds=de, attention_mask=da).logits[0,-1,:]
            ls = self._lm(inputs_embeds=se, attention_mask=sa).logits[0,-1,:]
            lb = self._lm(inputs_embeds=be, attention_mask=ba).logits[0,-1,:]
        return ld, ls, lb

    def generate_greedy(self, image, prompt, max_new_tokens=512):
        return self.generate_configured(
            image, prompt, max_new_tokens=max_new_tokens, do_sample=False,
        )

    def generate_configured(self, image, prompt, max_new_tokens=512,
                            do_sample=False, temperature=1.0,
                            top_p=1.0, top_k=None):
        """Generate with the decode policy registered by the experiment.

        The HF and LAVIS backends expose sampling through different keyword
        names, so normalize the shared pipeline interface here.
        """
        if self._backend == "hf":
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            vision_dtype = next(self.model.vision_model.parameters()).dtype
            inputs["pixel_values"] = inputs["pixel_values"].to(
                dtype=vision_dtype
            )
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
                out = self.model.generate(**inputs, **generation_kwargs)
            return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

        if top_k is not None:
            raise NotImplementedError(
                "The LAVIS InstructBLIP backend does not expose top-k sampling"
            )
        img_t = self.vis_processor(image).unsqueeze(0).to(self.device)
        generation_kwargs = {
            "max_length": max_new_tokens,
            "use_nucleus_sampling": do_sample,
        }
        if do_sample:
            generation_kwargs.update({
                "temperature": temperature,
                "top_p": top_p,
            })
        out = self.model.generate(
            {"image": img_t, "prompt": prompt}, **generation_kwargs
        )
        return out[0].strip() if out else ""
