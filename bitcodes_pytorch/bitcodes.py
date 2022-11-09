from typing import Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, einsum, nn

""" Utils """


def to_bits(indices: Tensor, num_bits: int) -> Tensor:
    bitmask = 2 ** torch.arange(num_bits - 1, -1, -1)
    return indices.unsqueeze(-1).bitwise_and(bitmask).ne(0).long()


def to_decimal(bits: Tensor) -> Tensor:
    num_bits = bits.shape[-1]
    bitmask = 2 ** torch.arange(num_bits - 1, -1, -1)
    return torch.sum(bitmask * bits, dim=-1)


""" Bincodes """


class Bitcodes(nn.Module):
    def __init__(self, features: int, num_bits: int, temperature: int):
        super().__init__()
        self.temperature = temperature
        self.codebook = nn.Parameter(torch.randn(2 * num_bits, features))

    def from_bits(self, bits: Tensor) -> Tensor:
        attn = F.one_hot(bits.long(), num_classes=2).float()
        attn = rearrange(attn, "b m p q -> b m (p q)")
        out = einsum("b m n, n d -> b m d", attn, self.codebook)
        return out

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        sim = einsum("b m d, n d -> b m n", x, self.codebook)
        pairs = rearrange(sim, "b m (p q) -> b m p q", q=2)

        if self.training:
            attn = F.gumbel_softmax(pairs, tau=self.temperature, dim=-1, hard=True)
        else:
            attn = F.one_hot(pairs.argmax(dim=-1), num_classes=2).float()

        attn = rearrange(attn, "b m p q -> b m (p q)")
        out = einsum("b m n, n d -> b m d", attn, self.codebook)
        bits = pairs.argmax(dim=-1)
        return out, bits
