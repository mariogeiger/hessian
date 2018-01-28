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
        hess = (torch.cuda.FloatTensor if output.is_cuda else torch.FloatTensor)(n, n).fill_(0)

    ai = 0
    for i, inp in enumerate(inputs):
        grad = torch.autograd.grad(output, inp, create_graph=True)
        grad = grad[0].contiguous().view(-1)

        for j in range(inp.numel()):
            row = torch.autograd.grad(grad[j], inputs[i:], retain_graph=True)
            row = torch.cat([x.data.contiguous().view(-1) for x in row])[j:]

            hess[ai, ai:] += row
            if ai + 1 < n:
                hess[ai + 1:, ai] += row[1:]
            del row
            ai += 1
        del grad

    return hess
