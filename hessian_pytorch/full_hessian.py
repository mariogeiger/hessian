# pylint: disable=E1101, C0103, C0111
'''
Highest eigenvalues and eigenvectors using the power method
'''
import torch


def full_hessian(fun, loader, parameters):
    '''
    Compute the Hessian of FUN with respect to `parameters`,

    FUN = sum of fun(batch) for batch in loader
    '''
    if next(iter(parameters)).is_cuda:
        Tensor = torch.cuda.FloatTensor
    else:
        Tensor = torch.FloatTensor

    n = sum(p.numel() for p in parameters)
    hessian = Tensor(n, n)

    ii = 0
    for j, param in enumerate(parameters):
        for i in range(param.numel()):
            row = [Tensor(p.size()).fill_(0) for p in parameters[j:]]

            for batch in loader:
                grad = torch.autograd.grad(fun(batch), param, create_graph=True)[0]
                grad = torch.autograd.grad(grad.view(-1)[i], parameters[j:])
                row = [x + y.data for x, y in zip(row, grad)]

            row = torch.cat([x.view(-1) for x in row])
            hessian[ii + i, ii:] = row
            hessian[ii:, ii + i] = row
            print("{}/{}      ".format(ii + i, n), end='\r')

        ii += param.numel()

    return hessian
