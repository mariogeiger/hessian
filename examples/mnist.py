#pylint: disable=C
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from hessian_pytorch import power_method

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def main(cuda):
    dataset = torchvision.datasets.MNIST('../data', train=True, download=True,
                                          transform=torchvision.transforms.Compose([
                                              torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                          ]))

    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=1, pin_memory=True)

    model = Net()
    if cuda:
        model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(loader):
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(loader.dataset),
                    100. * batch_idx / len(loader), loss.data[0]))

    for epoch in range(1, 10 + 1):
        train(epoch)


    loader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False, num_workers=1, pin_memory=True)

    def loss_function(batch):
        model.eval()  # disable dropout
        data, target = batch
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        output = model(data)
        loss = F.nll_loss(output, target, size_average=False) / len(dataset)
        return loss

    parameters = [p for p in model.parameters() if p.requires_grad]

    eigens = []

    while len(eigens) < 5:
        lam, vec = power_method(loss_function, loader, parameters, [vec for lam, vec in eigens],
                                target_overlap=0.999, min_iter=20, offset=0)
        eigens.append((lam, vec))
        print("the {}th evalue is {}".format(len(eigens), lam))
        print()


main(torch.cuda.is_available())
