## Extension of NALU with sine and cosine gate

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


class ExtNALU(nn.Module):
    def __init__(self, in_features: int, out_features: int, eps=1e-10):
        super(ExtNALU, self).__init__()
        self.W_ha = Parameter(torch.Tensor(out_features, in_features))
        self.M_ha = Parameter(torch.Tensor(out_features, in_features))
        self.W_hm = Parameter(torch.Tensor(out_features, in_features))
        self.M_hm = Parameter(torch.Tensor(out_features, in_features))
        self.G = Parameter(torch.Tensor(out_features, in_features))
        self.Gs = Parameter(torch.Tensor(out_features, in_features))
        self.Gc = Parameter(torch.Tensor(out_features, in_features))

        self.reset_parameters()
        self.eps = eps

    def reset_parameters(self):
        init.kaiming_uniform_(self.W_ha, a=math.sqrt(5))
        init.kaiming_uniform_(self.M_ha, a=math.sqrt(5))
        init.kaiming_uniform_(self.W_hm, a=math.sqrt(5))
        init.kaiming_uniform_(self.M_hm, a=math.sqrt(5))
        init.kaiming_uniform_(self.G, a=math.sqrt(5))
        init.kaiming_uniform_(self.Gs, a=math.sqrt(5))
        init.kaiming_uniform_(self.Gc, a=math.sqrt(5))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        Wa1 = torch.tanh(self.W_ha)
        Wa2 = torch.sigmoid(self.M_ha)
        Wa = Wa1 * Wa2

        Wm1 = torch.tanh(self.W_hm)
        Wm2 = torch.sigmoid(self.M_hm)
        Wm = Wm1 * Wm2

        g = input.matmul(self.G.t())
        g = torch.sigmoid(g)

        gs = input.matmul(self.Gs.t())
        gs = torch.sigmoid(gs)

        gc = input.matmul(self.Gc.t())
        gc = torch.sigmoid(gc)

        a = input.matmul(Wa.t())
        m = torch.log(torch.abs(input) + self.eps)
        m = m.matmul(Wm.t())
        m = torch.exp(m)
        y = g * a + (1 - g) * m

        y = gs * torch.sin(y) + (1 - gs) * y
        y = gc * torch.cos(y) + (1 - gc) * y

        return y


class ExtNALUNet(nn.Module):
    def __init__(self, in_features, struct, out_features):
        super(ExtNALUNet, self).__init__()
        layers = [ExtNALU(in_features, struct[0])]
        for i in range(len(struct)-1):
            layers.append(ExtNALU(struct[i], struct[i+1]))

        layers.append(ExtNALU(struct[-1], out_features))
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
        lamb = 0
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
    nalu = ExtNALUNet(2, struct, 1)
    nalu = nalu.to(device)

    dataset = TestDatasetV1()
    nalu.train_model(DataLoader(dataset, batch_size=20, shuffle=True), max_epochs=100, lrate=0.001)
    dataset.test_model(nalu)





