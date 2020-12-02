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
        xtest = np.concatenate([[np.linspace(-2, 2, 1000)], [np.linspace(-2, 2, 1000)]], axis=0)
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