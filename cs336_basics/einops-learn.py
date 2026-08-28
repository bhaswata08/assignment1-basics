import torch
from einops import einsum, rearrange, reduce


class CS336TutorialEinsum:
    """
    link: https://www.youtube.com/watch?v=kuYAsz7zspQ&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV&index=2
    """

    @staticmethod
    def einops_einsum():
        """Matmul"""
        x = torch.ones(3, 4)
        y = torch.ones(4, 3)

        # Example 1: Matmul
        ## Old way
        z = x @ y

        ## New way
        z = einsum(x, y, "seq1 hidden, hidden seq2 -> seq1 seq2")
        z = einsum(x, y, "ij, jk -> ik")

        # Example 2: Matrix vector multiplication
        a = torch.ones(2, 3)
        b = torch.ones(
            3,
        )
        c = einsum(a, b, "ij, j -> i")

        # Example 3: Hadamard prod
        a = torch.ones(2, 3)
        b = torch.ones(2, 3)
        c = einsum(a, b, "ij, ij -> ij")

        # Example 4: Batch Matmul
        x = torch.ones(2, 3, 4)  # batch seq1 hidden
        y = torch.ones(2, 3, 4)  # batch seq2 hidden

        ## Old way
        z = x @ y.transpose(-2, -1)  # batch seq1 seq2

        ## New way
        z = einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")
        ## one more way
        z = einsum(x, y, "... seq1 hidden, ... seq2 hidden -> ... seq1 seq2")

    @staticmethod
    def einops_reduce():
        """sum, mean, max, min"""
        x = torch.ones(2, 3, 4)

        # Old way
        y = x.sum(dim=-1)

        # New way
        y = reduce(x, "... hidden -> ...", "sum")

        a = torch.ones(2, 3)
        c = reduce(a, "ij ->", "sum")

        # col sum
        a = torch.ones(3, 4)

        # old way
        c = a.sum(dim=0)

        # new way
        c = reduce(a, "ij -> j", "sum")

        # row sum
        a = torch.ones(3, 4)

        # old way
        c = a.sum(dim=1)

        # new way
        c = reduce(a, "ij -> i", "sum")

    @staticmethod
    def einops_dot():
        # Dot product
        a = torch.ones(
            3,
        )
        b = torch.ones(
            3,
        )
        c = einsum(a, b, "i, i ->")

        # Outer prod of of two vectors u and v is u @ v.T
        # Remember, pytorch doesnt convert 1d vectors to transpose
        a = torch.ones(
            3,
        )
        b = torch.ones(
            2,
        )

        c = einsum(a, b, "i, j -> ij")

        # Trace
        a = torch.ones(3, 3)
        c = a.trace()
        c = einsum(a, "ii ->")

        # tensor dot
        a = torch.ones(3, 4, 2)
        b = torch.ones(4, 3, 6)
        c = einsum(a, b, "ijk, jil -> kl")

    @staticmethod
    def einops_rearrange():
        x = torch.ones(3, 8)  # the 8 here represents (4,2) matrix that you would like to have multiplied with w

        w = torch.ones(4, 4)

        x = rearrange(x, "... (heads hidden1) -> ... heads hidden1", heads=2)  # Converts (3, (8)) -> (3, 2, 4)

        x = einsum(x, w, "... hidden1, hidden1 hidden2 -> ... hidden2")


class EinopsRocks:
    """
    link: https://einops.rocks/1-einops-basics/
    """

    def __init__(self) -> None:
        x = torch.ones(3, 4, 5, 6)
        y = x.transpose(0, 2, 3, 1)
        y = rearrange(x, "b c h w -> b h c w")
