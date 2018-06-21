# pylint: disable=E1101, C0103, C0111
'''
Computes the Hessian
'''
import torch


def hessian(output, inputs, hess=None):
    '''
    Compute the Hessian of `output` with respect to `inputs`
    '''
    n = sum(p.numel() for p in inputs)
    if hess is None:
        hess = output.new_zeros(n, n)

    ai = 0
    for i, inp in enumerate(inputs):
        grad = torch.autograd.grad(output, inp, create_graph=True)
        grad = grad[0].contiguous().view(-1)

        for j in range(inp.numel()):
            if grad[j].requires_grad:
                row = torch.autograd.grad(grad[j], inputs[i:], retain_graph=True, allow_unused=True)
                row = [x if x is not None else torch.zeros_like(y) for x, y in zip(row, inputs[i:])]
                row = torch.cat([x.detach().contiguous().view(-1) for x in row])[j:]
            else:
                row = grad[j].new_zeros(sum(x.numel() for x in inputs[i:]) - j)

            hess[ai, ai:] += row.type_as(hess)  # ai's row
            if ai + 1 < n:
                hess[ai + 1:, ai] += row[1:].type_as(hess)  # ai's column
            del row
            ai += 1
        del grad

    return hess
