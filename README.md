# hessian_pytorch

Install with
```
python setup.py install
```
For the usage, look at the example files.


## Full hessian
The complete hessian is computed naively assuming the commutativity of the derivatives.

```python
import torch
from hessian_pytorch import hessian

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

## Power method
With the power method you can compute the highest (in amplitude, -10 is higher than +2) eigenvalue and its eigenvector.
Assuming the hessian symmetric, by projecting onto the orthogonal space of the already found eigenvectors you can find the second highest, third highest and so on.
