"""Scaled dot-product attention and causal multi-head self-attention."""

from typing import override

import torch
from einops import einsum
from jaxtyping import Bool, Float
from torch import Tensor, nn

from cs336_basics.model.layers import RMSNorm, softmax


class SDPA(nn.Module):
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... keys d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """

    def __init__(self, d_k: int | None = None, qk_norm: bool = False) -> None:
        super().__init__()
        self.qk_norm = qk_norm
        if qk_norm:
            assert d_k is not None, "d_k must be specified if qk_norm is to be enabled"
            self.q_norm = RMSNorm(d_k)
            self.k_norm = RMSNorm(d_k)

    @override
    def forward(
        self,
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys d_k"],
        V: Float[Tensor, " ... keys d_v"],
        mask: Bool[Tensor, "  ... queries keys"] | None = None,
    ) -> Float[Tensor, " ... queries d_v"]:
        d_k = Q.shape[-1]
        if self.qk_norm:
            Q, K = self.q_norm(Q), self.k_norm(K)

        scores = einsum(Q, K, " ... queries d_k, ... keys d_k -> ... queries keys") / (d_k**0.5)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = softmax(scores, dim=-1)
        return einsum(weights, V, " ... queries keys, ... keys d_v -> ... queries d_v")


if __name__ == "__main__":
    Q = torch.arange(100).reshape((1, 10, 10))
    print("Q shape:", Q.shape)
    K = torch.arange(100).reshape((1, 10, 10))
    V = torch.arange(100).reshape((1, 10, 10))
    SDPA().forward(Q, K, V)
