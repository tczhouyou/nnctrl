## the code is based on the paper: https://arxiv.org/pdf/1806.07259.pdf

import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.set_default_dtype(torch.float64)


class DivUnit(nn.Module):
    def __init__(self, in_features, out_features, theta=0.5):
        super(DivUnit, self).__init__()
        self.linear = nn.Linear(in_features, 2 * out_features)
        self.dim = out_features
        self.theta = theta

    def forward(self, x):
        x = x.to(device)
        y = self.linear(x)
        y1 = y[:, :self.dim]
        y2 = y[:, self.dim:]
        res = (y2 > self.theta) * (torch.div(y1,y2))
        return res, y2


class EQL(nn.Module):
    def __init__(self, in_feats, struct, out_feats):
        super(EQL, self).__init__()
        self.struct = struct
        linears = []
        in_features = in_feats
        for i in range(np.shape(struct)[0]):
            out_features = np.sum(struct[i,:-1]) + 2 * struct[i,-1]
            linears.append(nn.Linear(in_features, out_features))
            in_features = np.sum(struct[i,:])

        self.n_hlayers = np.shape(struct)[0]
        self.out_feats = out_feats
        self.in_feats = in_feats
        self.out = DivUnit(in_features, out_feats)
        self.linears = nn.ModuleList(linears)

    def apply_hidden_ops(self, inputs, layer_struct):
        cum_struct = np.cumsum(layer_struct)
        outputs = torch.ones([inputs.shape[0], cum_struct[-1]])
        outputs[:,cum_struct[0]: cum_struct[1]] = torch.sin(inputs[:,cum_struct[0]: cum_struct[1]])
        outputs[:,cum_struct[1]: cum_struct[2]] = torch.cos(inputs[:,cum_struct[1]: cum_struct[2]])
        y_multi = inputs[:,cum_struct[2]:]
        dim = layer_struct[-1]
        outputs[:, cum_struct[2]:] = y_multi[:, :dim] * y_multi[:, dim:]
        return outputs

    def forward(self, x):
        out = x
        for i in range(np.shape(self.struct)[0]):
            out = self.linears[i](out)
            out = self.apply_hidden_ops(out, self.struct[i,:])

        y_pred, y2 = self.out.forward(out)
        return y_pred, y2


def train_model(eql, dataloader, max_epochs=1000, lrate=0.001):
    t1 = 0.25 * max_epochs
    t2 = (19 / 20) * max_epochs

    mse = nn.MSELoss()
    optimizer = optim.Adam(eql.parameters(), lr=lrate)
    lamb = 1e-5
    for t in range(max_epochs):
        theta = 1 / np.sqrt(t + 1)
        eql.out.theta = theta

        epoch_loss = 0
        count = 0
        for i, (xt, yt) in enumerate(dataloader):
            if t > t2:
                for m_id in range(eql.n_hlayers):
                    eql.linears[m_id].weight = nn.Parameter((torch.abs(eql.linears[m_id].weight) > 0.001) * eql.linears[m_id].weight)

                eql.out.linear.weight = nn.Parameter((torch.abs(eql.out.linear.weight) > 0.001) * eql.out.linear.weight)

            optimizer.zero_grad()
            y_pred, y2 = eql.forward(xt)
            rep_loss = mse(yt, y_pred)
            pen_div = (theta - y2 > 0) * (theta - y2)
            pen_div = torch.sum(pen_div)
            if t1 < t < t2:
                l1_norm = 0
                for param in eql.parameters():
                    l1_norm += torch.sum(torch.abs(param))

                loss = rep_loss + pen_div + lamb * l1_norm
            else:
                loss = rep_loss + pen_div

            loss.backward()
            optimizer.step()
            epoch_loss += loss
            count += 1

        print('epoch: %d, loss: %.5f' % (t, epoch_loss / count))


class TestDataset(Dataset):
    # experiment in the paper: sin(pi * x_1) / (x_2^2 + 1)
    def __init__(self):
        self.X = np.random.uniform(-1,1,size=(10000,2))
        y = np.divide(np.sin(3.1415 * self.X[:,0]), np.power(self.X[:,1], 2) + 1)
        self.y = np.expand_dims(y, -1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = torch.from_numpy(self.X[index,:]).to(device)
        y = torch.from_numpy(self.y[index,:]).to(device)
        return x, y


if __name__ == "__main__":
    struct = np.array([[10,10,10,10]])
    eql = EQL(2, struct, 1)
    eql = eql.to(device)

    dataset = TestDataset()
    dataloader = DataLoader(dataset, batch_size=20, shuffle=True)
    train_model(eql, dataloader, max_epochs=40, lrate=0.001)

    xtest = np.concatenate([[np.linspace(-6,6,1000)],[np.linspace(-6,6,1000)]], axis=0)
    xtest = np.transpose(xtest)
    ytest = np.divide(np.sin(3.1415 * xtest[:,0]), np.power(xtest[:,1], 2) + 1)
    with torch.no_grad():
        y_pred, _ = eql.forward(torch.from_numpy(xtest).to(device))
        y_pred = y_pred.cpu()
        plt.plot(xtest[:,0], ytest, 'k-')
        plt.plot(xtest[:,0], y_pred[:,0], 'r-.')

    plt.show()




