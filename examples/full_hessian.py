#pylint: disable=C, E1101
import torch
from torch.autograd import Variable
from hessian_pytorch import full_hessian
import numpy as np

def main(cuda):
    x = torch.rand(128)
    if cuda:
        x = x.cuda()

    x = Variable(x, requires_grad=True)

    hessian = full_hessian(lambda _: torch.sum(x ** 2), [None], [x])
    evalues, evectors = np.linalg.eigh(hessian.cpu().numpy())

    print(evalues)

main(torch.cuda.is_available())
