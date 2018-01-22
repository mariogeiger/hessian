# pylint: disable=E1101, C0103, C0111
'''
Highest eigenvalues and eigenvectors using the power method
'''
import torch


def list_dot(a, b):
    return sum([torch.sum(x * y) for x, y in zip(a, b)])


def split_like(flat, lst):
    a = []
    i = 0
    for element in lst:
        n = element.numel()
        a.append(flat[i: i + n].view(*element.size()))
        i += n
    return a


def list_flatten(a):
    return torch.cat([x.view(-1) for x in a])


def full_hessian(fun, loader, parameters):
    '''
    Compute the Hessian of FUN with respect to `parameters`,

    FUN = sum of fun(batch) for batch in loader
    '''

    def hessian_vector_product(vector):
        vector = split_like(vector, parameters)

        # zero_grad
        for p in parameters:
            p.grad = None

        # computes dot(Hessian, vector)
        for batch in loader:
            grad = torch.autograd.grad(fun(batch), parameters, create_graph=True)
            list_dot(grad, [torch.autograd.Variable(x) for x in vector]).backward()
            del grad

        result = [p.grad.data for p in parameters]
        return list_flatten(result)

    n = sum(p.numel() for p in parameters)

    hessian = torch.FloatTensor(n, n)
    hot_vector = torch.zeros(n)
    if next(iter(parameters)).is_cuda:
        hessian = hessian.cuda()
        hot_vector = hot_vector.cuda()

    for i in range(n):
        hot_vector[i] = 1
        row = hessian_vector_product(hot_vector)
        hessian[i] = row
        hot_vector[i] = 0
        print("{}/{}      ".format(i, n), end='\r')

    return hessian
