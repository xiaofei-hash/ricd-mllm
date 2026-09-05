"""
R-CCR Risk-Calibrated Decoder

Module 4: Combines all modules into a single decoding step.
P_tilde(w) ∝ P_deep(w) * exp(-λ_l * R_l(w) + λ_s * R_s(w)) * I_APC(w)
"""

import torch
import torch.nn.functional as F
import json
import os
from typing import Optional, Dict, Any, Sequence

from ..proxy.conflict_proxies import ConflictProxies
from ..risk.risk_surrogates import RiskSurrogates


class RCCRDecoder:
    """
    Risk-Calibrated Cross-modal Conflict Resolution Decoder.

    At each decoding step:
    1. Get logits from deep/shallow/blind paths
    2. Compute L(w), V(w) proxies
    3. Compute R_l(w), R_s(w) risk surrogates with CSV gate
    4. Apply risk-calibrated reweighting with APC filter
    5. Select token from calibrated distribution
    """

    def __init__(
            self,
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
            binary_token_variants: Optional[Sequence[str]] = None,
            binary_margin_threshold: float = 0.0,
    ):
        self.lambda_l = lambda_l
        self.lambda_s = lambda_s
        self.proxies = ConflictProxies(tau_l=tau_l, tau_v=tau_v, top_k=top_k_proxy)
        self.risk = RiskSurrogates(theta_apc=theta_apc, apc_mode=apc_mode)
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive or None")
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.log_dir = log_dir
        self.preserve_binary_tokens_first_step = bool(
            preserve_binary_tokens_first_step
        )
        self.binary_token_variants = tuple(
            binary_token_variants
            or ("Yes", " yes", "yes", " Yes", "No", " no", "no", " No")
        )
        self.binary_margin_threshold = float(binary_margin_threshold)
        if self.binary_margin_threshold < 0:
            raise ValueError("binary_margin_threshold must be non-negative")
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def _binary_token_groups(self, tokenizer, vocab_size=None):
        """Resolve tokenizer-specific single-token Yes/No variants."""
        groups = {"yes": [], "no": []}
        if tokenizer is None:
            return groups
        for variant in self.binary_token_variants:
            label = variant.strip().lower()
            if label not in groups:
                continue
            token_ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(token_ids) != 1:
                continue
            token_id = int(token_ids[0])
            if vocab_size is None or 0 <= token_id < vocab_size:
                groups[label].append(token_id)
        return {key: sorted(set(value)) for key, value in groups.items()}

    def _preserve_binary_apc_tokens(self, apc_mask, tokenizer):
        """Restore single-token Yes/No variants to the step-0 APC support."""
        groups = self._binary_token_groups(tokenizer, apc_mask.numel())
        preserved = []
        for token_ids in groups.values():
            for token_id in token_ids:
                apc_mask[token_id] = 1.0
                preserved.append(token_id)
        return sorted(set(preserved))

    def decode_step(
        self,
        logits_deep: torch.Tensor,
        logits_shallow: torch.Tensor,
        logits_blind: torch.Tensor,
        step_id: int = 0,
        sample_id: str = "",
        tokenizer=None,
    ) -> Dict[str, Any]:
        """
        Single decoding step of R-CCR.

        Args:
            logits_deep: [vocab_size] logits from deep (full visual) path
            logits_shallow: [vocab_size] logits from shallow (early-exit visual) path
            logits_blind: [vocab_size] logits from blind (text-only) path
            step_id: Current decoding step index
            sample_id: Sample identifier for logging
            tokenizer: Optional tokenizer for logging token strings

        Returns:
            dict with selected_token_id, P_tilde, and debug info
        """
        device = logits_deep.device

        # Module 1: Cross-Modal Conflict Proxies
        L, V = self.proxies.compute(logits_deep, logits_shallow, logits_blind)

        # Module 2 & 3: Risk Surrogates + Safety Controls
        R_l, R_s, csv_gate, apc_mask = self.risk.compute_all(L, V, logits_deep)
        binary_apc_preserved_ids = []
        if self.preserve_binary_tokens_first_step and step_id == 0:
            binary_apc_preserved_ids = self._preserve_binary_apc_tokens(
                apc_mask, tokenizer
            )

        # Module 4: Risk-Calibrated Decoding
        # P_tilde(w) ∝ P_deep(w) * exp(-λ_l * R_l(w) + λ_s * R_s(w)) * I_APC(w)
        log_probs_deep = F.log_softmax(logits_deep, dim=-1)

        # Calibration in log-space
        log_calibration = -self.lambda_l * R_l + self.lambda_s * R_s
        log_p_tilde = log_probs_deep + log_calibration

        # Apply APC as a hard mask.  A tiny non-zero probability is unsafe for
        # multinomial decoding because an excluded token could still be drawn.
        log_p_tilde = log_p_tilde.masked_fill(apc_mask <= 0, float("-inf"))

        binary_margin_before_threshold = None
        binary_margin_after_threshold = None
        binary_yes_token_id = None
        binary_no_token_id = None
        if step_id == 0 and tokenizer is not None:
            groups = self._binary_token_groups(tokenizer, log_p_tilde.numel())
            yes_ids = groups["yes"]
            no_ids = groups["no"]
            if yes_ids and no_ids:
                yes_scores = log_p_tilde[yes_ids]
                no_scores = log_p_tilde[no_ids]
                yes_value, yes_index = torch.max(yes_scores, dim=0)
                no_value, no_index = torch.max(no_scores, dim=0)
                if torch.isfinite(yes_value) and torch.isfinite(no_value):
                    binary_yes_token_id = int(yes_ids[int(yes_index.item())])
                    binary_no_token_id = int(no_ids[int(no_index.item())])
                    binary_margin_before_threshold = float(
                        (yes_value - no_value).item()
                    )
                    if self.binary_margin_threshold > 0:
                        log_p_tilde = log_p_tilde.clone()
                        log_p_tilde[yes_ids] -= self.binary_margin_threshold
                    binary_margin_after_threshold = (
                        binary_margin_before_threshold
                        - self.binary_margin_threshold
                    )

        sampling_logits = log_p_tilde / self.temperature
        sampling_logits = self._apply_top_k_top_p(
            sampling_logits, top_k=self.top_k, top_p=self.top_p,
        )
        p_tilde = F.softmax(sampling_logits, dim=-1)

        if self.do_sample:
            selected_token_id = torch.multinomial(p_tilde, num_samples=1).item()
        else:
            selected_token_id = torch.argmax(p_tilde).item()

        # Also get baseline selection (without R-CCR)
        baseline_token_id = torch.argmax(F.softmax(logits_deep, dim=-1)).item()

        result = {
            "selected_token_id": selected_token_id,
            "baseline_token_id": baseline_token_id,
            "p_tilde": p_tilde,
            "changed": selected_token_id != baseline_token_id,
            "selection_mode": "multinomial" if self.do_sample else "greedy",
            "binary_apc_preserved_ids": binary_apc_preserved_ids,
            "binary_margin_threshold": self.binary_margin_threshold,
            "binary_margin_before_threshold": binary_margin_before_threshold,
            "binary_margin_after_threshold": binary_margin_after_threshold,
            "binary_yes_token_id": binary_yes_token_id,
            "binary_no_token_id": binary_no_token_id,
        }

        # Logging
        if self.log_dir and tokenizer is not None:
            self._log_step(
                logits_deep, logits_blind, logits_shallow,
                L, V, R_l, R_s, csv_gate, apc_mask,
                selected_token_id, baseline_token_id,
                step_id, sample_id, tokenizer
            )

        return result

    @staticmethod
    def _apply_top_k_top_p(logits, top_k=None, top_p=1.0):
        filtered = logits.clone()
        if top_k is not None and top_k < filtered.numel():
            cutoff = torch.topk(filtered, top_k).values[-1]
            filtered = filtered.masked_fill(filtered < cutoff, float("-inf"))
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
            cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cumulative > top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            remove_original = torch.zeros_like(remove).scatter(
                0, sorted_indices, remove,
            )
            filtered = filtered.masked_fill(remove_original, float("-inf"))
        return filtered

    def _log_step(self, logits_deep, logits_blind, logits_shallow,
                  L, V, R_l, R_s, csv_gate, apc_mask,
                  selected_id, baseline_id,
                  step_id, sample_id, tokenizer):
        """Save per-step debug log."""
        top_k = 10
        probs_deep = F.softmax(logits_deep, dim=-1)
        topk_vals, topk_ids = torch.topk(probs_deep, top_k)

        log_entry = {
            "sample_id": sample_id,
            "step_id": step_id,
            "selected_token_before": tokenizer.decode([baseline_id]),
            "selected_token_after": tokenizer.decode([selected_id]),
            "changed": selected_id != baseline_id,
            "top_tokens_deep": [
                {
                    "token": tokenizer.decode([tid.item()]),
                    "P_deep": round(probs_deep[tid].item(), 6),
                    "L": round(L[tid].item(), 4),
                    "V": round(V[tid].item(), 4),
                    "R_l": round(R_l[tid].item(), 4),
                    "R_s": round(R_s[tid].item(), 4),
                    "CSV": round(csv_gate[tid].item(), 1),
                    "APC": round(apc_mask[tid].item(), 1),
                }
                for tid in topk_ids
            ],
        }

        log_path = os.path.join(self.log_dir, f"{sample_id}.jsonl")
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


class BaselineDecoder:
    """Baseline greedy decoder (Regular)."""

    def decode_step(self, logits_deep: torch.Tensor, **kwargs) -> Dict[str, Any]:
        p = F.softmax(logits_deep, dim=-1)
        selected = torch.argmax(p).item()
        return {"selected_token_id": selected, "p_tilde": p, "changed": False}
