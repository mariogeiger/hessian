# pylint: disable=E1101, C0103, C0111
'''
Computes the gradient
'''
import torch


def gradient(output, inputs, retain_graph=True, create_graph=False):
    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)
    grads = torch.autograd.grad(output, inputs, allow_unused=True,
                                retain_graph=retain_graph,
                                create_graph=create_graph)
    grads = [x if x is not None else torch.zeros_like(y) for x, y in zip(grads, inputs)]
    return torch.cat([x.contiguous().view(-1) for x in grads])
