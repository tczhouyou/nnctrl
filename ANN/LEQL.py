# LEQL: Lagrangian EQL
# Based on EQL and Lagrangian network (https://arxiv.org/abs/2003.04630)
import os, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, "..")

import torch
import torch.nn as nn
from torch import optim
import numpy as np
from torch.utils.data import DataLoader
from DataSets import Pendulum
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.set_default_dtype(torch.float64)

## caculate jacobian and hessian (code source: https://gist.github.com/apaszke/226abdf867c4e9d6698bd198f3b45fb7)
def jacobian(y, x):
    jacob = []
    for i in range(y.shape[0]):
        bgrad = []
        for j in range(y.shape[1]):
            grad,  = torch.autograd.grad(y[i, j], x, retain_graph=True, create_graph=True)
            bgrad.append(grad[i,:])

        bgrad = torch.stack(bgrad)
        jacob.append(bgrad)

    return torch.squeeze(torch.stack(jacob), dim=1)

def hessian(y, x0, x1):
    hess = []
    for i in range(y.shape[0]):
        bgrad = []
        for j in range(y.shape[1]):
            grad, = torch.autograd.grad(y[i, j], x0, retain_graph=True, create_graph=True)
            grad = grad[i,:]
            dgrad = []
            for k in range(grad.shape[0]):
                dgradk, = torch.autograd.grad(grad[k], x1, retain_graph=True, create_graph=True)
                dgrad.append(dgradk[i,:])

            dgrad = torch.stack(dgrad)
            bgrad.append(dgrad)

        bgrad = torch.stack(bgrad)
        hess.append(bgrad)

    return torch.squeeze(torch.stack(hess), dim=1)

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


class LEQL(nn.Module):
    def __init__(self, dim, struct):
        super(LEQL, self).__init__()
        self.struct = struct
        linears = []
        in_features = dim * 2
        for i in range(np.shape(struct)[0]):
            out_features = np.sum(struct[i, :-1]) + 2 * struct[i, -1]
            linears.append(nn.Linear(in_features, out_features))
            in_features = np.sum(struct[i, :])

        self.n_hlayers = np.shape(struct)[0]
        self.out_feats = 1
        self.out = DivUnit(in_features, 1)
        self.linears = nn.ModuleList(linears)
        self.dim = dim

    def apply_hidden_ops(self, inputs, layer_struct):
        cum_struct = np.cumsum(layer_struct)
        outputs = torch.ones([inputs.shape[0], cum_struct[-1]])
        outputs[:, cum_struct[0]: cum_struct[1]] = torch.sin(inputs[:, cum_struct[0]: cum_struct[1]])
        outputs[:, cum_struct[1]: cum_struct[2]] = torch.cos(inputs[:, cum_struct[1]: cum_struct[2]])
        y_multi = inputs[:, cum_struct[2]:]
        dim = layer_struct[-1]
        outputs[:, cum_struct[2]:] = y_multi[:, :dim] * y_multi[:, dim:]
        return outputs

    def forward(self, x):
        q = x[:, :self.dim].clone().detach().requires_grad_(True).to(device)
        qd = x[:, self.dim:].clone().detach().requires_grad_(True).to(device)

        out = torch.cat([q,qd], dim=1)
        for i in range(np.shape(self.struct)[0]):
            out = self.linears[i](out)
            out = self.apply_hidden_ops(out, self.struct[i, :])
            out = out.to(device)

        lg, y2 = self.out.forward(out)
        jcob = jacobian(lg, q)
        hmat1 = hessian(lg, qd, qd)
        hmat2 = hessian(lg, q, qd)
        Imat = torch.eye(hmat1.shape[1]).reshape((1,hmat1.shape[1], hmat1.shape[1])).repeat(hmat1.shape[0], 1, 1)
        Imat = Imat.to(device)
        invMat = torch.inverse(hmat1 + 0.001 * Imat)
        jcobvec = torch.unsqueeze(jcob, -1) - torch.bmm(hmat2, torch.unsqueeze(q,-1))
        qdd = torch.matmul(invMat,jcobvec)
        return torch.squeeze(qdd, -1), lg, y2

    def train_model(self, dataloader, max_epochs=1000, lrate=0.001):
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

                optimizer.zero_grad()
                yt = yt.to(device)
                y_pred, lg, y2 = self.forward(xt)
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

            print('epoch: %d, loss: %.5f' % (t, epoch_loss / count))


if __name__ == "__main__":
    struct = np.array([[10,10,10,10]])
    leql = LEQL(1, struct)
    leql = leql.to(device)

    dataset = Pendulum()
    leql.train_model(DataLoader(dataset, batch_size=20, shuffle=True), max_epochs=200, lrate=0.0001)
    dataset.test_model(leql)

