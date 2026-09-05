"""
Dual Risk Surrogates and Safety Controls

Module 2 & 3 of R-CCR:
- R_l(w) = L(w) * (1 - V(w))  : language over-dominance risk
- R_s(w) = I_CSV(w) * (1 - L(w)) * V(w) : visual suppression risk
- CSV: Competitive Substitution Verification gate
- APC: Adaptive Plausibility Constraint
"""

import torch
import torch.nn.functional as F


class RiskSurrogates:
    """Compute R_l, R_s, CSV gate, and APC filter."""

    def __init__(self, theta_apc: float = 0.30,
                 apc_mode: str = "relative_max"):
        """
        Args:
            theta_apc: APC threshold.  In ``relative_max`` mode this is beta
                in P_deep(w) >= beta * max(P_deep); in ``absolute`` mode it
                is a fixed probability threshold.
            apc_mode: ``relative_max`` (paper/legacy Table 4) or ``absolute``
                (kept only for reproducing historical modular-code runs).
        """
        if apc_mode not in {"relative_max", "absolute"}:
            raise ValueError(
                f"Unsupported apc_mode={apc_mode!r}; expected relative_max or absolute"
            )
        if not 0.0 <= theta_apc <= 1.0:
            raise ValueError("theta_apc must be in [0, 1]")
        self.theta_apc = theta_apc
        self.apc_mode = apc_mode

    def compute_R_l(self, L: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Language over-dominance risk.
        R_l(w) = L(w) * (1 - V(w))
        High when: strong language prior + weak visual support
        """
        return L * (1.0 - V)

    def compute_csv_gate(self, R_l: torch.Tensor, logits_deep: torch.Tensor) -> torch.Tensor:
        """
        Competitive Substitution Verification gate.
        I_CSV(w) = 1 if R_l(w) < R_l(w*), else 0
        where w* = argmax P_deep(w)

        Only tokens safer than the current winner get suppression compensation.
        """
        probs_deep = F.softmax(logits_deep, dim=-1)
        w_star = torch.argmax(probs_deep, dim=-1)
        R_l_star = R_l[w_star]
        csv_gate = (R_l < R_l_star).float()
        return csv_gate

    def compute_R_s(self, L: torch.Tensor, V: torch.Tensor,
                    csv_gate: torch.Tensor) -> torch.Tensor:
        """
        Visual suppression risk (gated).
        R_s(w) = I_CSV(w) * (1 - L(w)) * V(w)
        High when: weak language prior + strong visual support + CSV pass
        """
        return csv_gate * (1.0 - L) * V

    def compute_apc_mask(self, logits_deep: torch.Tensor) -> torch.Tensor:
        """
        Adaptive Plausibility Constraint.
        relative_max: I_APC(w) = 1 if P_deep(w) >= beta * max P_deep
        absolute:     I_APC(w) = 1 if P_deep(w) >= theta_apc
        """
        probs_deep = F.softmax(logits_deep, dim=-1)
        if self.apc_mode == "relative_max":
            cutoff = self.theta_apc * probs_deep.max(dim=-1, keepdim=True).values
        else:
            cutoff = torch.as_tensor(
                self.theta_apc, dtype=probs_deep.dtype, device=probs_deep.device
            )
        apc_mask = (probs_deep >= cutoff).float()

        # Defensive invariant: APC must never remove the deep-path winner.
        winner = probs_deep.argmax(dim=-1, keepdim=True)
        apc_mask.scatter_(-1, winner, 1.0)
        return apc_mask

    def compute_all(self, L: torch.Tensor, V: torch.Tensor,
                    logits_deep: torch.Tensor):
        """
        Compute all risk surrogates and safety controls.

        Returns:
            R_l, R_s, csv_gate, apc_mask
        """
        R_l = self.compute_R_l(L, V)
        csv_gate = self.compute_csv_gate(R_l, logits_deep)
        R_s = self.compute_R_s(L, V, csv_gate)
        apc_mask = self.compute_apc_mask(logits_deep)
        return R_l, R_s, csv_gate, apc_mask
