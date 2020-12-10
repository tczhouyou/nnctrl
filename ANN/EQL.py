## the code is based on the paper: https://arxiv.org/pdf/1806.07259.pdf

import os, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, "..")

import os
import click
import torch
import torch.nn as nn
from torch import optim
import numpy as np
from torch.utils.data import DataLoader
from DataSets import Pendulum, TestDatasetV1, TestEQLDataset
import matplotlib.pyplot as plt
from utils import create_path

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
        out = x.clone().detach().to(device)
        for i in range(np.shape(self.struct)[0]):
            out = self.linears[i](out)
            out = self.apply_hidden_ops(out, self.struct[i,:])
            out = out.to(device)

        y_pred, y2 = self.out.forward(out)
        return y_pred, y2

    def train_model(self, dataloader, model_path, max_epochs=1000, lrate=0.001):
        t1 = 0.25 * max_epochs
        t2 = (19 / 20) * max_epochs

        mse = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lrate)
        lamb = 1e-5
        for t in range(max_epochs):
            theta = 1 / np.sqrt(t + 1)
            self.out.theta = theta

            epoch_loss = 0
            count = 0
            for i, (xt, yt) in enumerate(dataloader):
                if t > t2:
                    for m_id in range(self.n_hlayers):
                        self.linears[m_id].weight = nn.Parameter((torch.abs(self.linears[m_id].weight) > 0.001) * self.linears[m_id].weight)

                    self.out.linear.weight = nn.Parameter((torch.abs(self.out.linear.weight) > 0.001) * self.out.linear.weight)

                yt = yt.to(device)
                optimizer.zero_grad()
                y_pred, y2 = self.forward(xt)
                rep_loss = mse(yt, y_pred)
                pen_div = (theta - y2 > 0) * (theta - y2)
                pen_div = torch.sum(pen_div)
                if t1 < t < t2:
                    l1_norm = 0
                    for param in self.parameters():
                        l1_norm += torch.sum(torch.abs(param))

                    loss = rep_loss + pen_div + lamb * l1_norm
                else:
                    loss = rep_loss + pen_div

                loss.backward()
                optimizer.step()
                epoch_loss += loss
                count += 1

            if t % 50:
                torch.save(self.state_dict(), model_path)

            print('epoch: %d, loss: %.5f' % (t, epoch_loss / count))

    def test_model(self, dataset, model_path):
        self.load_state_dict(torch.load(model_path))
        with torch.no_grad():
            y_pred = self.forward(torch.from_numpy(dataset.X).to(device))
            if isinstance(y_pred, tuple):
                y_pred = y_pred[0]

            y_pred = y_pred.cpu()
            x = np.arange(y_pred.shape[0])

            for i in range(y_pred.shape[1]):
                plt.figure()
                plt.plot(x, dataset.Y[:, i], 'k-')
                plt.plot(x, y_pred[:, i], 'r-.')
                print(((y_pred[:, i] - dataset.Y[:, i]) ** 2).mean())
            plt.show()


@click.command()
@click.option("--train", default=0, help="1: training, 0: testing")
def main(train):
    struct = np.array([[10,10,10,10]])
    eql = EQL(2, struct, 1)
    eql = eql.to(device)

    dataset = TestDatasetV1()
    model_path = "../saved_model/test_eql_v1"
    create_path(model_path)
    model_path = os.path.join(model_path, "state_dict_model.pt")
    if train:
        eql.train_model(DataLoader(dataset, batch_size=20, shuffle=True), model_path=model_path,  max_epochs=100, lrate=0.001)
    else:
        test_dataset = TestEQLDataset()
        eql.test_model(test_dataset, model_path)


if __name__ == "__main__":
    main()

