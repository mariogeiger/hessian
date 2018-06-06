# hessian_pytorch

Install with
```
python setup.py install
```
For the usage, look at the example files.


## Full hessian
The complete hessian is computed naively assuming the commutativity of the derivatives.

## Power method
With the power method you can compute the highest (in amplitude, -10 is higher than +2) eigenvalue and its eigenvector.
Assuming the hessian symmetric, by projecting onto the orthogonal space of the already found eigenvectors you can find the second highest, third highest and so on.
