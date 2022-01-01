from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
    """ Simple convnet for classification. """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1   = nn.Linear(320, 50)
        self.fc2   = nn.Linear(50, 10)

    def forward(self, x):
        h = F.relu(F.max_pool2d(self.conv1(x), 2))
        h = F.relu(F.max_pool2d(self.conv2(h), 2))
        h = h.view(-1, 320)
        h = F.relu(self.fc1(h))
        h = self.fc2(h)
        return F.log_softmax(h, dim=1)
