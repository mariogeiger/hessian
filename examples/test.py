#pylint: disable=C, E1101, E1102
import torch
from hessian import hessian

def main(device):
    x = torch.rand(4, requires_grad=True, device=device)
    x1 = x[0].pow(2) + x[1].pow(2)
    x2 = x[2].pow(2) + x[3].pow(2)
    y = x1 * x2

    h = hessian(y, x)
    print(h)

    h_exact = torch.tensor([
        [2 * x2, 0, 4 * x[0] * x[2], 4 * x[0] * x[3]],
        [0, 2 * x2, 4 * x[1] * x[2], 4 * x[1] * x[3]],
        [4 * x[0] * x[2], 4 * x[1] * x[2], 2 * x1, 0],
        [4 * x[0] * x[3], 4 * x[1] * x[3], 0, 2 * x1],
    ])
    print(h_exact)

    e, v = torch.symeig(h, eigenvectors=True)
    print(e)
    print(v)


main(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

