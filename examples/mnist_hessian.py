# pylint: disable = C, R, E1101
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import hessian_pytorch
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(10, 10, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(4 * 10, 10)

    def forward(self, x):  # pylint: disable = W0221
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 4 * 10)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


def train(model, dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=1, pin_memory=torch.cuda.is_available())

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    def make_epoch(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(loader):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(loader.dataset),
                    100. * batch_idx / len(loader), loss.item()))

    for epoch in range(1, 2 + 1):
        make_epoch(epoch)


def test(model, dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False, num_workers=1, pin_memory=torch.cuda.is_available())

    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    print("accuracy = {}".format(correct / len(loader.dataset)))


def compute_hessian(model, dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=2000, shuffle=True, num_workers=1, pin_memory=torch.cuda.is_available())

    parameters = [p for p in model.parameters() if p.requires_grad]
    n = sum(p.numel() for p in parameters)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hessian = torch.zeros(n, n, device=device)

    for i, (data, target) in enumerate(loader):
        if i >= 2:  # TODO remove
            break

        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = F.nll_loss(output, target, size_average=False) / len(dataset)

        hessian_pytorch.hessian(loss, parameters, hess=hessian)
        print('{}/{}    '.format(i, len(loader)), end='\r')


    evalues, evectors = np.linalg.eigh(hessian.cpu().numpy())
    print(evalues[-20:])
    print(evectors[:, 0])


def main():
    torch.backends.cudnn.benchmark = True

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = torchvision.datasets.MNIST('../data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST('../data', train=False, transform=transform)

    model = Net()
    if torch.cuda.is_available():
        model.cuda()

    train(model, trainset)
    test(model, testset)
    compute_hessian(model, trainset)


main()
