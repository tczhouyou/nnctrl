import torch
import torch.nn as nn

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.set_default_dtype(torch.float64)


class FNN(nn.Module):
    def __init__(self, n_layers, act="leaky_relu"):
        super(FNN, self).__init__()
        layers = []
        for i in range(len(n_layers)-1):
            in_ = n_layers[i]
            out_ = n_layers[i+1]
            layers.append(nn.Linear(in_, out_))
            if act == "relu":
                layers.append(nn.ReLU())
            elif act == "leaky_relu":
                layers.append(nn.LeakyReLU())

        self.fnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.fnn(x)




