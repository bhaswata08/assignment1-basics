import math
from typing import override

import torch
from einops import einsum, rearrange, reduce, repeat
from jaxtyping import Float, Int
from torch import Tensor, nn

from cs336_basics.transformers.utils import check_positive


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        check_positive(in_features=in_features, out_features=out_features)

        std: float = math.sqrt(2 / (in_features + out_features))
        self.weight: Float[nn.Parameter, " out_features in_features"] = nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(size=(out_features, in_features), dtype=dtype, device=device),
                mean=0,
                std=std,
                a=-3 * std,
                b=3 * std,
            )
        )

    @override
    def forward(self, x: Float[Tensor, " ... in_features"]) -> Float[Tensor, " ... out_features"]:
        return einsum(self.weight, x, " out_features in_features, ... in_features -> ... out_features")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        check_positive(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.weight: Float[nn.Parameter, " num_embeddings embedding_dim"] = nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(
                    size=(num_embeddings, embedding_dim),
                    dtype=dtype,
                    device=device,
                ),
                mean=0,
                std=1,
                a=-3,
                b=3,
            )
        )

    @override
    def forward(self, x: Int[Tensor, " ... seq_len"]) -> Float[Tensor, " ... seq_len _"]:
        return self.weight[x]


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        check_positive(d_model=d_model, eps=eps)
        self.weight: Float[nn.Parameter, " d_model"] = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype),
        )
        self.d_model: int = d_model
        self.eps: float = eps

    @override
    def forward(self, x: Float[Tensor, " ... seq_len d_model"]) -> Float[Tensor, " ... seq_len d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(reduce(x**2, "... seq_len d_model -> ... seq_len", "sum") / self.d_model + self.eps)
        rms_broadcast = rearrange(rms, "... seq_len -> ... seq_len 1")
        result = x / rms_broadcast * self.weight
        return result.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        raw = int(8 / 3 * d_model)
        d_ff_tmp = math.ceil(raw / 64) * 64 if not d_ff else d_ff
        # d_ff_tmp: int = int(((8 / 3 * d_model) * 64) // 64) if not d_ff else d_ff

        check_positive(d_model=d_model, d_ff=d_ff_tmp)
        self.w1: Linear = Linear(in_features=d_model, out_features=d_ff_tmp, device=device, dtype=dtype)
        self.w2: Linear = Linear(in_features=d_ff_tmp, out_features=d_model, device=device, dtype=dtype)
        self.w3: Linear = Linear(in_features=d_model, out_features=d_ff_tmp, device=device, dtype=dtype)

    @staticmethod
    def silu(x: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
        return x * torch.sigmoid(x)

    @override
    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        gate: Float[Tensor, " ... d_ff_tmp"] = self.silu(self.w1(x))
        up: Float[Tensor, " ... d_ff_tmp"] = self.w3(x)
        gated: Float[Tensor, " ... d_ff_tmp"] = einsum(gate, up, " ... d_ff_tmp, ... d_ff_tmp -> ... d_ff_tmp")
        out: Float[Tensor, " ... d_model"] = self.w2(gated)
        return out


class RoPE(nn.Module):
    # rope_emb: Float[Tensor, " max_seq_len d_k"]

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        # Guide to RoPE: https://www.youtube.com/watch?v=V8r__fXx7tU
        check_positive(theta=theta, d_k=d_k, max_seq_len=max_seq_len)
        rope_emb = self.precompute_rope_embeddings(
            d_k=d_k, max_seq_len=max_seq_len, theta=theta, device=device, dtype=dtype
        )
        self.register_buffer("rope_emb", rope_emb, persistent=False)

    @staticmethod
    def precompute_rope_embeddings(
        d_k: int, max_seq_len: int, theta: float, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> Float[Tensor, " max_seq_len d_k"]:
        if d_k % 2 != 0:
            raise ValueError("RoPE dimension d_k should be divisible by 2")

        sub_dk: int = d_k // 2

        i: Float[Tensor, " max_seq_len"] = torch.arange(
            0, max_seq_len, device=device, dtype=dtype
        ).float()  # idx for iterating thru seq len
        k: Float[Tensor, " sub_dk"] = torch.arange(
            0, d_k // 2, device=device, dtype=dtype
        ).float()  # idx for pairing up d_k and iterating through
        inv_freq: Float[Tensor, " sub_dk"] = 1.0 / theta ** ((2 * k) / d_k)  # 0 indexed hence no (2*k -2)
        base_freq = einsum(i, inv_freq, " max_seq_len, sub_dk -> max_seq_len sub_dk")

        emb = repeat(base_freq, "max_seq_len sub_dk -> max_seq_len (sub_dk 2)")
        return emb

    @staticmethod
    def apply_rotarty_pos_emb(
        x: Float[Tensor, " ... seq_len d_k"],
        rope_emb: Float[Tensor, " max_seq_len d_k"],
        token_positions: Int[Tensor, " ... seq_len"],
    ) -> Float[Tensor, " ... seq_len d_k"]:
        # seq_len: int = x.shape[-2]  # [ ... seq_len d_k] -> seq_len
        # rope_emb_sliced = rope_emb[
        #     :seq_len, :
        # ]  # [max_seq_len d_k] -> [seq_len d_K] ## x may or may not be upto max_seq_len, therefore to avoid unnecessary computation we only calc to :seq_len of x

        token_positions = token_positions.to(x.device)

        rope_emb_sliced = rope_emb[
            token_positions
        ]  # use the token_positions instead to account for autoregressive decoding instead of assuming it counts from 0 idx

        cos = rope_emb_sliced.cos()
        sin = rope_emb_sliced.sin()

        # Split the last dimension d_k into pairs
        x_pairs = rearrange(x.float(), " ... seq_len (d_k pair) -> ... seq_len d_k pair", pair=2)
        x0, x1 = x_pairs[..., 0], x_pairs[..., 1]

        x_partner = rearrange([-x1, x0], " pair ... seq_len d -> ... seq_len (d pair)")

        return (x.float() * cos + x_partner * sin).type_as(x)

    @override
    def forward(
        self, x: Float[Tensor, " ... seq_len d_k"], token_positions: Int[Tensor, " ... seq_len"]
    ) -> Float[Tensor, " ... seq_len d_k"]:
        assert isinstance(self.rope_emb, torch.Tensor)
        return self.apply_rotarty_pos_emb(x, self.rope_emb, token_positions)


def softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    max_val = torch.amax(
        in_features, dim=dim, keepdim=True
    )  # Equivalent to reduce(" ... c -> ... 1", "max") but since einops doesnt support dynamic dims we are using amax
    exp_tensor = torch.exp(in_features - max_val)  # subtracted to avoid numerical stability
    return exp_tensor / torch.sum(exp_tensor, dim=dim, keepdim=True)


if __name__ == "__main__":
    rope = RoPE(theta=10, d_k=16, max_seq_len=6)
