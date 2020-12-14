# Interaction Network. The code is based on https://arxiv.org/pdf/1612.00222.pdf and https://github.com/jsikyoon/Interaction-networks_tensorflow

import os, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, "..")

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from DataSets import ThreeBodyProblem
from ANN.basic_models import FNN
from ANN.EQL import EQL
from torch import optim

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.set_default_dtype(torch.float64)

class INN(nn.Module):
    def __init__(self, num_nodes, num_rels, effect_dim, act_dim, rel_net, obj_net, sys, batch_size=20):
        super(INN, self).__init__()
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.rel_net = rel_net
        self.obj_net = obj_net
        self.effect_dim = effect_dim
        self.act_dim = act_dim
        self.sys = sys

    def mfunc(self, O, Ra):
        batch_size = O.shape[0]
        num_nodes = O.shape[-1]
        num_rels = Ra.shape[-1]
        self.Rr = torch.zeros(batch_size, num_nodes, num_rels).to(device)
        self.Rs = torch.zeros(batch_size, num_nodes, num_rels).to(device)

        cnt = 0
        for i in range(num_nodes):
            for j in range(num_nodes):
                if (i != j):
                    self.Rr[:, i, cnt] = 1.0
                    self.Rs[:, j, cnt] = 1.0
                    cnt += 1

        return torch.cat((torch.matmul(O, self.Rr), torch.matmul(O, self.Rs), Ra), 1)

    def step(self, states, rels, ext_effs):
        # states: batch_size x state_size x num_nodes
        # rels: batch_size x rel_feat_size x num_rels
        # ext_effs: batch_size x ext_feat_size x num_nodes
        Bmat = self.mfunc(states, rels)
        Bmat = torch.transpose(Bmat, 1, 2)
        Bmat = torch.reshape(Bmat, (-1, Bmat.shape[2]))
        e_t = self.rel_net.forward(Bmat)
        e_t = torch.reshape(e_t, (-1, self.num_rels, self.effect_dim))
        e_t = torch.transpose(e_t, 1, 2)
        Rr_trans = torch.transpose(self.Rr, 1, 2)
        e_bar = torch.matmul(e_t, Rr_trans)
        c_agg = torch.cat((states, ext_effs, e_bar), 1)
        c_agg = torch.transpose(c_agg, 1, 2)
        c_agg = torch.reshape(c_agg, (-1, c_agg.shape[2]))
        ostates = self.obj_net.forward(c_agg)
        ostates = torch.reshape(ostates, (-1, self.num_nodes * self.act_dim))
        # ostates = torch.transpose(ostates, 1, 2)
        return ostates

    def forward(self, x):
        if callable(getattr(self.sys, 'get_inputs')):
            states, rels, ext_effs = self.sys.get_inputs(x)
            states = torch.from_numpy(states).to(device)
            rels = torch.from_numpy(rels).to(device)
            ext_effs = torch.from_numpy(ext_effs).to(device)
            return self.step(states, rels, ext_effs)

    def train_model(self, model_path='test', max_epochs=1000, lrate=0.001):
        dataloader = DataLoader(self.sys, batch_size=20, shuffle=True)
        mse = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lrate)
        for t in range(max_epochs):
            epoch_loss = 0
            count = 0
            for i, (xt, yt) in enumerate(dataloader):
                yt = yt.to(device)
                optimizer.zero_grad()
                y_pred = self.forward(xt)
                loss = mse(yt, y_pred)
                loss.backward()
                optimizer.step()
                epoch_loss += loss
                count += 1

            if t % 50:
                torch.save(self.state_dict(), model_path)

            print('epoch: %d, loss: %.5f' % (t, epoch_loss / count), end='\r')


def three_body():
    num_nodes = 3
    num_rels = 9
    effect_dim = 5

    ## fixed
    state_size = 2
    rel_feat_size = 1
    ext_effect_size = 1
    act_dim = 2

    tb_sys = ThreeBodyProblem()
    obj_net = FNN([state_size + ext_effect_size + effect_dim, 40, 40, act_dim])
    obj_net.to(device)
    rel_net = FNN([2 * state_size+rel_feat_size, 40, 40, effect_dim])
    rel_net.to(device)

    inn = INN(num_nodes, num_rels, effect_dim, act_dim, rel_net, obj_net, tb_sys)
    inn.train_model(max_epochs=12000, lrate=0.0001)

    vig8vel = np.array([-0.9324, -0.8647])
    y0 = np.array([[0.97, -0.243, -0.97, 0.243, 0, 0]])
    dy0 = np.array([[-vig8vel[0]/2, -vig8vel[1]/2, -vig8vel[0]/2, -vig8vel[1]/2, vig8vel[0], vig8vel[1]]])
    tb_sys.test_model(inn, y0, dy0, T=10000)


if __name__ == "__main__":
    three_body()







