from torch.utils.data import Dataset
import numpy as np
import torch
import matplotlib.pyplot as plt
from abc import abstractmethod
device = torch.device("cuda:0" if  torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

class TestDatasetV1(Dataset):
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

    def test_model(self, model):
        xtest = np.concatenate([[np.linspace(-6, 6, 1000)], [np.linspace(-6, 6, 1000)]], axis=0)
        xtest = np.transpose(xtest)
        ytest = np.divide(np.sin(3.1415 * xtest[:, 0]), np.power(xtest[:, 1], 2) + 1)
        with torch.no_grad():
            y_pred = model.forward(torch.from_numpy(xtest).to(device))
            if isinstance(y_pred, tuple):
                y_pred = y_pred[0]

            y_pred = y_pred.cpu()
            plt.plot(xtest[:, 0], ytest, 'k-')
            plt.plot(xtest[:, 0], y_pred[:, 0], 'r-.')

        plt.show()


class TestEQLDataset(Dataset):
    def __init__(self):
        x = np.concatenate([[np.linspace(-6, 6, 1000)], [np.linspace(-6, 6, 1000)]], axis=0)
        self.X = np.transpose(x)
        self.Y = np.expand_dims(np.divide(np.sin(3.1415 * self.X[:, 0]), np.power(self.X[:, 1], 2) + 1), axis=1)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        x = torch.from_numpy(self.X[index, :]).to(device)
        y = torch.from_numpy(self.Y[index, :]).to(device)
        return x, y

class Pendulum(Dataset):
    def __init__(self, l=1, g=10):
        super(Pendulum, self).__init__()
        x = np.concatenate([np.random.uniform(-np.pi/3, np.pi/3, size=(1000,1)), np.random.uniform(-np.pi/2, np.pi/2, size=(1000,1))], axis=1)
        print(np.shape(x))
        self.a = g / l
        self.l = l
        self.g = g
        y = np.zeros(shape=(1000, 1))
        for i in range(np.shape(x)[0]):
            th = x[i, 0]
            th_dd = - self.a * np.sin(th)
            y[i] = th_dd

        self.X = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = self.X[index,:]
        y = self.y[index,:]
        return x, y

    def test_model(self, model):
        xtest = np.concatenate([[np.linspace(-np.pi, np.pi, 100)], [np.linspace(-np.pi, np.pi, 100)]], axis=0)
        xtest = np.transpose(xtest)
        ytest = - self.a * np.sin(xtest[:,0])
        ltest = 0.5 * self.l * self.l * np.power(xtest[:,1], 2) - self.g * self.l * (1 - np.cos(xtest[:,0]))
        xtest = torch.from_numpy(xtest)
        res = model.forward(xtest)
        if isinstance(res, tuple):
            y_pred = res[0]
            l_pred = res[1]
            l_pred = l_pred.detach().cpu()
            plt.plot(xtest[:, 0], ltest, 'k-')
            plt.plot(xtest[:, 0], l_pred[:, 0], 'g-.')

        y_pred = y_pred.detach().cpu()
        plt.plot(xtest[:, 0], ytest, 'k-')
        plt.plot(xtest[:, 0], y_pred[:, 0], 'r-.')
        plt.show()


class CountDataset(Dataset):
    # experiment: counting
    def __init__(self):
        self.X = np.random.uniform(0,10,size=(50,1))
        self.y = self.X + 1

    def __getitem__(self, index):
        x = self.X[index, :]
        y = self.y[index, :]
        return x, y

    def __len__(self):
        return len(self.y)

    def test_model(self, model):
        xtest = np.array([np.linspace(0, 20, 20)])
        xtest = np.transpose(xtest)
        print(xtest)
        ytest = xtest + 1
        with torch.no_grad():
            y_pred = model.forward(torch.from_numpy(xtest).to(device))
            if isinstance(y_pred, tuple):
                y_pred = y_pred[0]

            y_pred = y_pred.cpu()
            plt.plot(xtest[:, 0], ytest, 'k-')
            plt.plot(xtest[:, 0], y_pred[:, 0], 'r-.')

        plt.show()


class DynamicSystem(Dataset):
    def __init__(self):
        super(DynamicSystem, self).__init__()
        self.dt = 0.005
        self.method = 'euler'

    def step(self, y, dy):
        ddy = self.dyn(y)
        dy = dy + self.dt * ddy
        y = y + self.dt * dy
        return y, dy

    @abstractmethod
    def dyn(self, y):
        raise NotImplementedError()

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

class ThreeBodyProblem(DynamicSystem):
    def __init__(self, wsize=10, m1=10, m2=10, m3=10, G=6.674):
        super(ThreeBodyProblem, self).__init__()
        self.y = np.random.uniform(0, wsize, size=(1000, 12))
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.G = G
        self.dy = self.dyn(self.y)

    def dyn(self, y):
        r1 = y[:, 0:2]
        r2 = y[:, 2:4]
        r3 = y[:, 4:6]

        dy = np.zeros(shape=(np.shape(y)[0], 6))
        dy[:, 0:2] = - self.G * self.m2 * (r1 - r2) / np.power(np.linalg.norm(r1 - r2, axis=1, keepdims=True), 3) \
                     - self.G * self.m3 * (r1 - r3) / np.power(np.linalg.norm(r1 - r3, axis=1, keepdims=True), 3)
        dy[:, 2:4] = - self.G * self.m3 * (r2 - r3) / np.power(np.linalg.norm(r2 - r3, axis=1, keepdims=True), 3) \
                      - self.G * self.m1 * (r2 - r1) / np.power(np.linalg.norm(r2 - r1, axis=1, keepdims=True), 3)
        dy[:, 4:6] = - self.G * self.m1 * (r3 - r1) / np.power(np.linalg.norm(r3 - r1, axis=1, keepdims=True), 3) \
                       - self.G * self.m2 * (r3 - r2) / np.power(np.linalg.norm(r3 - r2, axis=1, keepdims=True), 3)

        return dy

    def __getitem__(self, index):
        return self.y[index, :6], self.dy[index, :6]

    def __len__(self):
        return len(self.y)

    def get_inputs(self, y):
        batch_size = np.shape(y)[0]
        states = np.zeros(shape=(batch_size, 2, 3))
        rels = np.zeros(shape=(batch_size, 1, 9))
        ext_effs = np.zeros(shape=(batch_size, 1, 3))

        states[:, :, 0] = y[:, 0:2]
        states[:, :, 1] = y[:, 2:4]
        states[:, :, 2] = y[:, 4:6]

        for i in range(3):
            for j in range(3):
                rels[:, :, 3*i+j] = np.linalg.norm(states[:, :, i] - states[:, :, j], axis=1, keepdims=True)

        return states, rels, ext_effs

    def test_model(self, model, y0, dy0, T=100):
        yg = y0
        dyg = dy0
        yt = y0
        dyt = dy0

        gb1 = y0[:,0:2]
        gb2 = y0[:,2:4]
        gb3 = y0[:,4:6]

        for i in range(T):
            # x = torch.from_numpy(yt[:, :6]).to(device)
            # ddy = model.forward(x)
            # dyt = dyt + self.dt * ddy
            # yt = yt + self.dt * dyt

            ## ground truth
            yg, dyg = self.step(yg, dyg)

            gb1 = np.concatenate([gb1, yg[:,0:2]], axis=0)
            gb2 = np.concatenate([gb2, yg[:,2:4]], axis=0)
            gb3 = np.concatenate([gb3, yg[:,4:6]], axis=0)

            plt.cla()
            plt.plot(gb1[:,0], gb1[:,1], 'r-')
            plt.plot(gb2[:,0], gb2[:,1], 'g-')
            plt.plot(gb3[:,0], gb3[:,1], 'b-')
            plt.axis([-5,5,-5,5])
            plt.pause(0.1)










