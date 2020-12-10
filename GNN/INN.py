import torch
import torch.nn as nn
from torch import optim
import numpy as np
from torch.utils.data import DataLoader

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.set_default_dtype(torch.float64)

class INN(nn.Module):
    def __init__(self, num_nodes, rel_func, obj_func):
        super(INN, self).__init__()
        self.num_nodes = num_nodes
        self.rel_func = rel_func
        self.obj_func = obj_func


