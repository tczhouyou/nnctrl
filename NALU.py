
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.set_default_dtype(torch.float64)


class NALU(nn.Module):
    def __init__(self, in_features: int, out_features: int, eps=1e-10):
        super(NALU, self).__init__()
        self.W_h = Parameter(torch.Tensor(out_features, in_features))
        self.M_h = Parameter(torch.Tensor(out_features, in_features))
        self.G = Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()
        self.eps = eps

    def reset_parameters(self):
        init.kaiming_uniform_(self.W_h, a=math.sqrt(5))
        init.kaiming_uniform_(self.M_h, a=math.sqrt(5))
        init.kaiming_uniform_(self.G, a=math.sqrt(5))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        W1 = torch.tanh(self.W_h)
        W2 = torch.sigmoid(self.M_h)
        W = W1 * W2
        g = input.matmul(self.G.t())
        g = torch.sigmoid(g)
        a = input.matmul(W.t())
        m = torch.log(torch.abs(input) + self.eps)
        m = m.matmul(W.t())
        m = torch.exp(m)
        y = g * a + (1 - g) * m
        return y


class NALUNet(nn.Module):
    def __init__(self, in_features, layers, out_features):
        super(NALUNet, self).__init__()
        self.layers = [NALU(in_features, layers[0])]
        for i in range(len(layers)-1):
            self.layers.append(NALU(layers[i], layers[i+1]))



