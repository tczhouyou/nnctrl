from torch.utils.data import Dataset
import numpy as np
import torch
import matplotlib.pyplot as plt

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



