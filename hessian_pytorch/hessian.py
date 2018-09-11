# pylint: disable=E1101, C0103, C0111
'''
Computes the Hessian
'''
import torch


def gradient(output, inputs, retain_graph=True, create_graph=False):
    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)
    grads = torch.autograd.grad(output, inputs, allow_unused=True, retain_graph=retain_graph, create_graph=create_graph)
    grads = [x if x is not None else torch.zeros_like(y) for x, y in zip(grads, inputs)]
    return torch.cat([x.contiguous().view(-1) for x in grads])


def hessian(output, inputs, hess=None, allow_unused=False, create_graph=False):
    '''
    Compute the Hessian of `output` with respect to `inputs`
    '''
    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)
    n = sum(p.numel() for p in inputs)
    if hess is None:
        hess = output.new_zeros(n, n)

    ai = 0
    for i, inp in enumerate(inputs):
        [grad] = torch.autograd.grad(output, inp, create_graph=True, allow_unused=allow_unused)
        grad = grad.contiguous().view(-1)

        for j in range(inp.numel()):
            if grad[j].requires_grad:
                row = gradient(grad[j], inputs[i:], create_graph=create_graph)[j:]
            else:
                row = grad[j].new_zeros(sum(x.numel() for x in inputs[i:]) - j)

            hess[ai, ai:].add_(row.type_as(hess))  # ai's row
            if ai + 1 < n:
                hess[ai + 1:, ai].add_(row[1:].type_as(hess))  # ai's column
            del row
            ai += 1
        del grad

    return hess
