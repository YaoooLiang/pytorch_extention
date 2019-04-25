import torch
import torch.nn as nn
from modules.add import MyAddModule


class Network(nn.Module):
    """docstring for Network"""
    def __init__(self):
        super(Network, self).__init__()
        self.add = MyAddModule()

    def forward(self, input1, input2):
        return self.add(input1, input2)


model = Network()
input1, input2 = torch.randn(5, 5), torch.randn(5, 5)
print(model(input1, input2))
print(input1 + input2)
