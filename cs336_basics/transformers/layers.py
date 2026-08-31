"""Linear, Embedding, RMSNorm"""

import math
from typing import override

import torch
from einops import einsum, reduce
from jaxtyping import Float, Int
from torch import Tensor, nn


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if in_features <= 0:
            raise ValueError("in_features must be postive")
        if out_features <= 0:
            raise ValueError("out_features must be postive")

        stdev = math.sqrt(2 / (in_features + out_features))

        self.W: Float[nn.Parameter, "out_features in_features"] = nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty((out_features, in_features), dtype=dtype, device=device),
                mean=0.0,
                std=stdev,
                a=-3 * stdev,
                b=3 * stdev,
            )
        )

    @override
    def forward(self, x: Float[Tensor, "... in_features"]) -> Float[Tensor, "... out_features"]:
        return einsum(x, self.W, "... in_features, out_features in_features -> ... out_features")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        num_embeddings: Size of the vocabulary
        embedding_dim: Dimension of the embedding vectors, i.e., 𝑑_model
        device: Device to store the parameters on
        dtype: Data type of the parameters
        """
        super().__init__()

        if num_embeddings <= 0:
            raise ValueError("Vocab size must be postive")
        if embedding_dim <= 0:
            raise ValueError("d_model must be postive")

        stdev = 1
        self.W: Float[nn.Parameter, "vocab_size d_model"] = nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty((num_embeddings, embedding_dim), dtype=dtype, device=device),
                mean=0.0,
                std=stdev,
                a=-3 * stdev,
                b=3 * stdev,
            )
        )

    @override
    def forward(self, x: Int[Tensor, "... seqlen"]) -> Float[Tensor, "... seqlen d_model"]:
        # x must of of the type Int or Long so that there are no floating point indices
        return self.W[x]


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        """Construct the RMSNorm module.

        Args:
            d_model (int): Hidden dimension of the model.
            eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-5.
            device (torch.device | None, optional): Device to store the parameters on. Defaults to None.
            dtype (torch.dtype | None, optional): Data type of the parameters. Defaults to None.
        """
        super().__init__()
        self.eps: float = eps
        self.weight: Float[nn.Parameter, d_model] = nn.Parameter(
            torch.ones((d_model), dtype=dtype, device=device),
        )

    @override
    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)  # Prevent overflow
        mean_sq = reduce(x**2, "... d_model -> ... 1", reduction=torch.mean)  # $\sum x^2$
        rsqrt = torch.rsqrt(mean_sq + self.eps)  # $\frac{1}{\sqrt{\sum x^2}}$
        result = x * rsqrt * self.weight
        return result.to(in_dtype)


if __name__ == "__main__":
    m = Linear(in_features=1000, out_features=10)
    emb = Embedding(num_embeddings=100, embedding_dim=64)
    x = torch.randint(0, 100, (2, 10))  # (batch, seq_len)
    out = emb(x)
    assert out.shape == (2, 10, 64)
    # Check init stats
    print(emb.W.mean(), emb.W.std())  # Should be ~0, ~1
    print(emb.W.min(), emb.W.max())  # Should be within [-3, 3]
