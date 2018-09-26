# pylint: disable=E1101, C0103, C0111
'''
Highest eigenvalues and eigenvectors using gradient ascent of the Rayleigh quotient
'''
import torch
from torch.autograd import Variable


def list_dot(a, b):
    return sum([torch.sum(x * y) for x, y in zip(a, b)])


def list_sum(a, b):
    return [x + y for x, y in zip(a, b)]


def list_sub(a, b):
    return [x - y for x, y in zip(a, b)]


def list_norm(a):
    return list_dot(a, a) ** 0.5


def list_normalize(a):
    nor = list_norm(a)
    return [x / nor for x in a]


def list_flatten(a):
    return torch.cat([x.view(-1) for x in a])


def rayleigh_quotient(fun, loader, parameters):
    parameters = list(parameters)
    vector = list_normalize([torch.rand(*p.size()) for p in parameters])

    if next(iter(parameters)).is_cuda:
        vector = [x.cuda() for x in vector]

    vector = [Variable(x, requires_grad=True) for x in vector]

    optimizer = torch.optim.Adam(vector)

    for _ in range(100):
        # Computes Hv
        for p in parameters:
            p.grad = None

        for batch in loader:
            grad = torch.autograd.grad(fun(batch), parameters, create_graph=True)
            list_dot(grad, [Variable(x.data) for x in vector]).backward()

        H_vector = [p.grad.data for p in parameters]

        # Compute the gradient of the Rayleigh quotient
        for x, hv in zip(vector, H_vector):
            x.grad = Variable(-hv + x.data * torch.sum(x.data * hv))
            # minus the gradient because the optimizer minimize

        # Optimize
        optimizer.step()

        # Normalize the vector
        n = list_norm([x.data for x in vector])
        for x in vector:
            x.data /= n

    return [x.data for x in vector]
