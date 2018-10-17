# hessian

Install with
```
python setup.py install
```

## Compute Hessian
```python
import torch
from hessian import hessian

x = torch.tensor([1.5, 2.5], requires_grad=True)
h = hessian(x.pow(2).prod(), x, create_graph=True)

print(h)
# tensor([[12.5, 15],
#         [15,  4.5]], grad_fn=<CopySlices>)

h2 = hessian(h.sum(), x)
print(h2)
# tensor([[4, 8],
#         [8, 4]])
```

The hessian is computed naively assuming the commutativity of the derivatives.

## Compute Jacobian
```python
import torch
from hessian import jacobian

x = torch.tensor([1.5, 2.5], requires_grad=True)
y = torch.tensor([5.5, -4.], requires_grad=True)
j = jacobian(x.pow(y), [x, y])

print(j)
# tensor([[34.1, -0.00,  3.77,  0.00],
#         [ 0.0, -0.04,  0.00,  0.02]])
```
