# pylint: disable=no-member, line-too-long
'''
Computes the Hessian
'''
import torch
from .gradient import gradient


def hessian(output, inputs, out=None, allow_unused=False, create_graph=False):
    r'''
    Compute the Hessian of `output` with respect to `inputs`
    ```
    hessian((x * y).sum(), [x, y])
    ```
    '''
    assert output.ndimension() == 0

    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)

    numel = sum(p.numel() for p in inputs)
    if out is None:
        out = output.new_zeros(numel, numel)

    row_index = 0
    for i, inp in enumerate(inputs):
        [grad] = torch.autograd.grad(output, inp, create_graph=True, allow_unused=allow_unused)
        grad = torch.zeros_like(inp) if grad is None else grad
        grad = grad.contiguous().view(-1)

        for j in range(inp.numel()):
            if grad[j].requires_grad:
                row = gradient(grad[j], inputs[i:], retain_graph=True, create_graph=create_graph)[j:]
            else:
                row = grad[j].new_zeros(sum(x.numel() for x in inputs[i:]) - j)

            out[row_index, row_index:].add_(row.type_as(out))  # row_index's row
            if row_index + 1 < numel:
                out[row_index + 1:, row_index].add_(row[1:].type_as(out))  # row_index's column
            del row
            row_index += 1
        del grad

    return out
