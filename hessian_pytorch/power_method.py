# pylint: disable=E1101, C0103, C0111
'''
Highest eigenvalues and eigenvectors using the power method
'''
import torch


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


def power_method(fun, loader, parameters, orthogonals, offset=0, target_overlap=0.9999, min_iter=1, max_iter=10**10):
    '''
    Compute the Hessian's eigenvector of FUN with respect to `parameters`,
    orthogonal to all vectors provided in `orthogonals`

    FUN = sum of fun(batch) for batch in loader
    '''
    parameters = list(parameters)
    vector = list_normalize([torch.rand(*p.size()) for p in parameters])

    if next(iter(parameters)).is_cuda:
        vector = [x.cuda() for x in vector]

    def project(vector):
        # project to orthogonal
        for orthogonal in orthogonals:
            dot = list_dot(vector, orthogonal)
            vector = list_sub(vector, [dot * x for x in orthogonal])
        return vector

    vector = project(vector)

    for niter in range(max_iter):
        # zero_grad
        for p in parameters:
            p.grad = None

        # hessian vector product
        # computes dot(Hessian + offset, vector)
        for batch in loader:
            grad = torch.autograd.grad(fun(batch), parameters, create_graph=True)
            list_dot(grad, [torch.autograd.Variable(x) for x in vector]).backward()

        lam_vector = [p.grad.data + offset * v for p, v in zip(parameters, vector)]

        # project to orthogonal
        lam_vector = project(lam_vector)

        # eigenvalue
        lam = list_dot(vector, lam_vector) / list_dot(vector, vector) - offset

        # overlap
        new_vector = list_normalize(lam_vector)
        overlap = abs(list_dot(vector, new_vector))
        vector = new_vector

        print('iter =', niter, 'current overlap =', overlap, 'target overlap =', target_overlap)

        if niter > min_iter and overlap > target_overlap:
            break

    return lam, vector
