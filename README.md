
# Bitcodes - PyTorch

A new vector quantization method with binary codes, in PyTorch.

```bash
pip install bitcodes-pytorch
```
[![PyPI - Python Version](https://img.shields.io/pypi/v/bitcodes-pytorch?style=flat&colorA=black&colorB=black)](https://pypi.org/project/bitcodes-pytorch/)


## Usage

### Quantize
```python
from bitcodes_pytorch import Bitcodes

bitcodes = Bitcodes(
    features=8,
    num_bits=4,
    temperature=10,
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

### Recover Output from Bits
```python
y_decoded = bitcodes.from_bits(bits)

assert torch.allclose(y, y_decoded) # Assert passes in eval mode!
```

### Decimal-Binary Conversion
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

TODO
