## the code is based on the paper: https://arxiv.org/pdf/1806.07259.pdf

import os, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert("..", current_dir)

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
from torch import optim
import numpy as np
from torch.utils.data import DataLoader
import math
from DataSets import TestDatasetV1

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
    def __init__(self, in_features, struct, out_features):
        super(NALUNet, self).__init__()
        layers = [NALU(in_features, struct[0])]
        for i in range(len(struct)-1):
            layers.append(NALU(struct[i], struct[i+1]))

        layers.append(NALU(struct[-1], out_features))
        self.n_layers = len(layers)
        self.layers = nn.ModuleList(layers)

    def forward(self, input):
        out = input
        for i in range(self.n_layers):
            out = self.layers[i](out)

        return out

    def train_model(self, dataloader, max_epochs=1000, lrate=0.001):
        mse = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lrate)
        lamb = 1e-10
        for t in range(max_epochs):
            epoch_loss = 0
            count = 0
            for i, (xt, yt) in enumerate(dataloader):
                optimizer.zero_grad()
                y_pred = self.forward(xt)
                rep_loss = mse(yt, y_pred)
                l1_norm = 0
                for param in self.parameters():
                    l1_norm += torch.sum(torch.abs(param))

                loss = rep_loss + lamb * l1_norm
                loss.backward()
                optimizer.step()
                epoch_loss += loss
                count += 1

            print('epoch: %d, loss: %.5f' % (t, epoch_loss / count))


if __name__ == "__main__":
    struct = np.array([100,100])
    nalu = NALUNet(2, struct, 1)
    nalu = nalu.to(device)

    dataset = TestDatasetV1()
    nalu.train_model(DataLoader(dataset, batch_size=20, shuffle=True), max_epochs=100, lrate=0.001)
    dataset.test_model(nalu)





