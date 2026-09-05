"""
Cross-Modal Conflict Proxies: L(w) and V(w)

Module 1 of R-CCR.
- L(w): Language dominance proxy — how strongly a token is driven by language prior
- V(w): Visual support proxy — how strongly a token is supported by deep visual conditioning

Critical design choice — τ controls sigmoid sharpness:
  smaller positive τ → sharper L,V responses
  larger positive τ  → smoother L,V responses
  fixed τ is selected by task/backbone; τ=0 enables per-step auto-calibration

Supports two modes:
  Fixed τ   : use provided tau_l, tau_v directly
  Auto τ    : set tau_l=0 or tau_v=0 → calibrate from per-step Δ distribution
"""

import torch
import torch.nn.functional as F


class ConflictProxies:
    """Compute L(w) and V(w) from three-path logits."""

    def __init__(self, tau_l: float = 0.3, tau_v: float = 0.3,
                 top_k: int = 10, auto_tau_percentile: float = 0.15):
        """
        Args:
            tau_l: Temperature for L(w) sigmoid.  0 = auto-calibrate each step.
            tau_v: Temperature for V(w) sigmoid.  0 = auto-calibrate each step.
            top_k: Number of top tokens for mu_topk in L(w).
            auto_tau_percentile: When auto, set τ = this × std(Δ) among top-200 tokens.
        """
        self.tau_l = tau_l
        self.tau_v = tau_v
        self.top_k = top_k
        self.auto_pct = auto_tau_percentile

    def _effective_tau(self, delta: torch.Tensor, fixed_tau: float) -> float:
        """Get τ — use fixed if > 0, else auto-calibrate from Δ distribution."""
        if fixed_tau > 0:
            return fixed_tau
        # Auto: τ = percentile × std of Δ among top-200 tokens (by |Δ|)
        _, top_idx = torch.topk(delta.abs(), min(200, delta.numel()))
        std = delta[top_idx].std().item()
        return max(std * self.auto_pct, 0.01)  # floor to avoid division by ~0

    def compute_L(self, logits_blind: torch.Tensor) -> torch.Tensor:
        """
        Language dominance proxy.
        L(w) = σ( Δ_l(w) / τ_l )
        where Δ_l(w) = log P_blind(w) - μ_topk(log P_blind)
        """
        log_p = F.log_softmax(logits_blind, dim=-1)

        topk_vals, _ = torch.topk(log_p, self.top_k, dim=-1)
        mu_topk = topk_vals.mean(dim=-1, keepdim=True)

        delta_l = log_p - mu_topk

        tau = self._effective_tau(delta_l, self.tau_l)
        L = torch.sigmoid(delta_l / tau)
        return L

    def compute_V(self, logits_deep: torch.Tensor,
                  logits_shallow: torch.Tensor) -> torch.Tensor:
        """
        Visual support proxy.
        V(w) = σ( Δ_v(w) / τ_v )
        where Δ_v(w) = log P_deep(w) - log P_shallow(w)
        """
        log_deep    = F.log_softmax(logits_deep, dim=-1)
        log_shallow = F.log_softmax(logits_shallow, dim=-1)

        delta_v = log_deep - log_shallow

        tau = self._effective_tau(delta_v, self.tau_v)
        V = torch.sigmoid(delta_v / tau)
        return V

    def compute(self, logits_deep, logits_shallow, logits_blind):
        """Compute both L(w) and V(w)."""
        L = self.compute_L(logits_blind)
        V = self.compute_V(logits_deep, logits_shallow)
        return L, V
