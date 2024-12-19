#!/usr/bin/env python
# coding: utf-8

# In[1]:

from scipy.stats import poisson

import torch
import math
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair as pair
from torch.autograd import Variable
from torch.nn import init

class DM_Dataset(Dataset):
    def __init__(self, spatial_tempoal_view, spatial_view, temporal_view, Y, coor_index, X_index):
        self.spatial_tempoal_view = Variable(torch.Tensor(spatial_tempoal_view).float())
        self.spatial_view = Variable(torch.Tensor(spatial_view).float())
        self.temporal_view = Variable(torch.Tensor(temporal_view).float())
        self.Y = Variable(torch.Tensor(Y).float())
        self.X_index = Variable(torch.Tensor(X_index).int())
        self.coor_index = Variable(torch.Tensor(coor_index).int())
        
        self.T_encd = T_concept_encoding(self.temporal_view)
        self.S_encd = S_concept_encoding(self.spatial_view)
        self.ST_encd = ST_concept_encoding(self.spatial_tempoal_view)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return (self.spatial_tempoal_view[idx], self.ST_encd[idx], self.T_encd[idx], self.S_encd[idx]), self.spatial_view[idx], self.temporal_view[idx], self.Y[idx], self.coor_index, self.X_index[idx]




