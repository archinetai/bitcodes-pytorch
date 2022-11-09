
# Bitcodes - PyTorch

A new vector quantization method with binary codes, in PyTorch.

```bash
pip install bitcodes-pytorch
```
[![PyPI - Python Version](https://img.shields.io/pypi/v/bitcodes-pytorch?style=flat&colorA=black&colorB=black)](https://pypi.org/project/bitcodes-pytorch/)


## Usage

```python
from bitcodes_pytorch import Bitcodes

bitcodes = Bitcodes(
    features=8,     # Number of features per vector
    num_bits=4,     # Number of bits per vector
    temperature=10, # Gumbel softmax training temperature
)

# Set to eval during inference to make deterministic
bitcodes.eval()

x = torch.randn(1, 6, 8)
# Computes y, the quantzed version of x, and the bitcodes
y, bits = bitcodes(x)

"""
y.shape = torch.Size([1, 6, 8])

bits = tensor([[
  [0, 0, 0, 0],
  [1, 0, 1, 1],
  [1, 0, 0, 1],
  [1, 0, 0, 0],
  [0, 1, 1, 1],
  [0, 0, 1, 0]
]])
"""
```

### Dequantize
```python
y_decoded = bitcodes.from_bits(bits)

assert torch.allclose(y, y_decoded) # Assert passes in eval mode!
```

### Utils: Decimal-Binary Conversion
```python
from bitcodes_pytorch import to_decimal, to_binary

indices = to_decimal(bits)
# tensor([[ 0, 11,  9,  8,  7,  2]])

bits = to_binary(indices, num_bits=4)

"""
bits = tensor([[
  [0, 0, 0, 0],
  [1, 0, 1, 1],
  [1, 0, 0, 1],
  [1, 0, 0, 0],
  [0, 1, 1, 1],
  [0, 0, 1, 0]
]])
"""
```

## Explaination

Current vector quantization methods (e.g. [VQ-VAE](https://arxiv.org/abs/1711.00937#), [RQ-VAE](https://arxiv.org/abs/2203.01941)) either use a single large codebook or multiple smaller codebooks that are used as residuals. Residuals allow for an exponential increase in the number of possible combinations while keeping the number of total codebook items reasonably small by overlapping many codebook elements. If we let $C$ be the codebook size, and $R$ the number of residuals, we can get a theoretical maximum of $C^R$ combinations, assuming that all residuals have the same codebook size. The total number of codebook elements, which is proportional to parameter count, is instead $C\cdot R$. Thus it makes sense to keep $C$ as small as possible to maintain the parameter count reasonably small, while increasing $R$ to exploit the exponential number of combinations.

Here we use $C=2$ making the code binary, where $R=$`num_bits` can be freely chosen. The residuals are overlapped to get the output, instead of quantizing the difference - this allows to remove the residual loop and quantize with large $R$ in parallel.

Another nice property of bitcodes is that we can choose to quantize the bit matrix to integers in different ways after training (e.g. convert to decimal one or two rows at a time).
