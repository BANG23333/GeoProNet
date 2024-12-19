#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import math
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import random 
import seaborn as sns
import matplotlib
import pickle
from sklearn.model_selection import train_test_split
import os
from sklearn.decomposition import LatentDirichletAllocation
import scipy
from scipy import stats
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from scipy.stats import poisson

import torch
import math
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair as pair
from torch.autograd import Variable
from torch.nn import init

class ProtoNet(nn.Module):
    def __init__(self):
        super(ProtoNet, self).__init__()
    
        # self.convlstm = EncoderDecoderConvLSTM(6, 6).to(device)
        # self.LSTM = nn.LSTM((12), hidden_size = 12,batch_first=True)
        
        self.S_proto_num = 8
        # self.ST_prototype_vectors = nn.Parameter(torch.rand(self.proto_num, 9*9*6), requires_grad=True)
        # self.T_prototype_vectors = nn.Parameter(torch.rand(self.proto_num, 12), requires_grad=True)
        # self.S_prototype_vectors = nn.Parameter(torch.rand(self.S_proto_num, wide*wide), requires_grad=True)
        self.S_prototype_vectors = nn.Parameter(torch.rand(self.S_proto_num, 998), requires_grad=True)
        
        self.loc_weights = nn.Parameter(torch.rand(49), requires_grad=True)

        self.qz_loga = Parameter(torch.Tensor(self.S_proto_num))
        # self.FC= nn.Linear(self.proto_num, 1, bias=False)
        # self.FC2= nn.Linear(self.proto_num, self.proto_num, bias=False)
        
        #self.FC3= nn.Linear(29, 1)
        #self.BN1= nn.BatchNorm1d(9*9*6)
        #self.BN2= nn.BatchNorm1d(9*9*6)
        self.soft = nn.Softmax(dim=1)
        self.FC= nn.Linear(self.S_proto_num, 2, bias=False)

        self.loc_m = nn.Sigmoid()
        # self.m1 = nn.Sigmoid()

        # self.L0 = L0Dense(860, 860, bias=True, weight_decay=1., droprate_init=0.5, temperature=2./3., lamba=1., local_rep=False).to(device)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        # self.m = nn.AvgPool2d(3, padding=1)
        self.ppd = nn.ZeroPad2d(1)
        
    def forward(self, spatial_tempoal_view, spatial_view, temporal_view):
        
        # print(spatial_tempoal_view.shape) # [8, 506, 600]
        # print(spatial_view.shape) # [8, 506, 1300]
        # print(temporal_view.shape) # 
        weights = self.loc_m(self.loc_weights)

        b, loc, _ = spatial_view.size()
        spatial_view = spatial_view.reshape(b, loc, 18, 74, 4)
        
        spatial_view_3 = spatial_view[:, :, :, :49, :]
        spatial_view_3 = spatial_view_3.reshape(b, loc, 18, 7, 7, 4)

        spatial_view_5 = spatial_view[:, :, :, 49:, :]
        spatial_view_5 = spatial_view_5.reshape(b, loc, 18, 5, 5, 4)

        spatial_view_5 = spatial_view_5.flatten(start_dim=0, end_dim=1).permute(0, 1, 4, 2, 3).flatten(start_dim=1, end_dim=2)
        spatial_view_5 = self.ppd(spatial_view_5)
        
        spatial_view_3 = spatial_view_3.flatten(start_dim=0, end_dim=1).permute(0, 1, 4, 2, 3).flatten(start_dim=1, end_dim=2)

        merged2 = torch.zeros(spatial_view_5.shape).to(device)

        spatial_view_5 = spatial_view_5.flatten(start_dim=2, end_dim=3)
        spatial_view_3 = spatial_view_3.flatten(start_dim=2, end_dim=3)
    
        merged2 = spatial_view_5*weights + (1-weights)*spatial_view_3
        merged2 = merged2.reshape(merged2.shape[0], merged2.shape[1], 7, 7)
        
        
        b, loc, _ = spatial_tempoal_view.size()
        
        spatial_tempoal_view = spatial_tempoal_view.reshape(b, loc, 4, 74, 4)

        spatial_tempoal_view_3 = spatial_tempoal_view[:, :, :, :49, :]
        spatial_tempoal_view_3 = spatial_tempoal_view_3.reshape(b, loc, 4, 7, 7, 4)

        spatial_tempoal_view_5 = spatial_tempoal_view[:, :, :, 49:, :]
        spatial_tempoal_view_5 = spatial_tempoal_view_5.reshape(b, loc, 4, 5, 5, 4)
        
        spatial_tempoal_view_5 = spatial_tempoal_view_5.flatten(start_dim=0, end_dim=1).permute(0, 1, 4, 2, 3).flatten(start_dim=1, end_dim=2)
        spatial_tempoal_view_5 = self.ppd(spatial_tempoal_view_5)
        
        spatial_tempoal_view_3 = spatial_tempoal_view_3.flatten(start_dim=0, end_dim=1).permute(0, 1, 4, 2, 3).flatten(start_dim=1, end_dim=2)

        merged = torch.zeros(spatial_tempoal_view_5.shape).to(device)

        spatial_tempoal_view_5 = spatial_tempoal_view_5.flatten(start_dim=2, end_dim=3)
        spatial_tempoal_view_3 = spatial_tempoal_view_3.flatten(start_dim=2, end_dim=3)
    
        merged = spatial_tempoal_view_3*weights + (1-weights)*spatial_tempoal_view_5
        merged = merged.reshape(merged.shape[0], merged.shape[1], 7, 7)
        

        temporal_view = temporal_view.unsqueeze(2).expand(-1, -1, spatial_tempoal_view.shape[1]).permute(0, 2, 1)
        
        
        spatial_view = avg_directional_pooling(merged2) # avg_directional_pooling
        spatial_tempoal_view = avg_directional_pooling(merged) # max_directional_pooling
        
        spatial_view = spatial_view.reshape(b*loc, -1)
        spatial_tempoal_view = spatial_tempoal_view.reshape(b*loc, -1)
        temporal_view = temporal_view.reshape(b*loc, -1)
        
        ST_encd, T_encd, S_encd = spatial_tempoal_view.clone().detach(), temporal_view.clone().detach(), spatial_view.clone().detach()
        
        output = torch.cat((temporal_view, spatial_view, spatial_tempoal_view), 1)

        # shad = self.S_prototype_vectors.clone()
        # shad -= shad.min(1, keepdim=True)[0]
        # shad /= shad.max(1, keepdim=True)[0]
        
        # topk, indices = torch.topk(shad, 128, dim = 1)
              
        # res = torch.zeros(self.S_prototype_vectors.shape).to(device)
        # res = res.scatter(1, indices, topk)
        # res = torch.where(res == 0, 0, 1)
        
        # self.S_prototype_vectors =  nn.Parameter(torch.mul(self.S_prototype_vectors, res), requires_grad=True)
                
        spatial_proto = 1/torch.cdist(output, self.S_prototype_vectors[None], p=2).squeeze(0)
        
        shad = spatial_proto.clone()
        shad -= shad.min(1, keepdim=True)[0]
        shad /= shad.max(1, keepdim=True)[0]
        spatial_proto = shad
        
        # spatial_proto = pearson_corr(output, self.S_prototype_vectors)     
        
        output = self.FC(spatial_proto)
        output = self.soft(output) # [32384, 3]

        # loss_diver_S = diversity_cost_spatial2(torch.mul(shad, res))
        # loss_diver_S = diversity_cost_spatial2(self.S_prototype_vectors)
        loss_l0 = self.L0(self.S_prototype_vectors)
        res = torch.ones(860).to(device)

        #shad = self.S_prototype_vectors.clone()
        #shad -= shad.min(1, keepdim=True)[0]
        #shad /= shad.max(1, keepdim=True)[0]
        
        """
        topk, indices = torch.topk(shad, 128, dim = 1)
        res = torch.zeros(self.S_prototype_vectors.shape).to(device)
        res = res.scatter(1, indices, topk)
        res = torch.where(res == 0, 0, 1)
        """
        
        # self.S_prototype_vectors -= self.S_prototype_vectors.min(1, keepdim=True)[0]
        # self.S_prototype_vectors /= self.S_prototype_vectors.max(1, keepdim=True)[0]

        loss_diver = 0
        ctr = 0

        for i in range(self.S_prototype_vectors.size()[0]):
            for j in range(self.S_prototype_vectors.size()[0]):
                if i == j:
                    continue
                ctr += 1
                loss_diver += torch.cdist(self.S_prototype_vectors[i].expand(1, -1), self.S_prototype_vectors[j].expand(1, -1), p=2)

        loss_diver = -1 * loss_diver/ctr
        
        
        return output, 0, spatial_proto, (ST_encd, T_encd, S_encd, res), loss_diver
    
    def L0(self, x):
        prior_prec,lamba = 1, 1
        logpw_col = torch.sum(- (.5 * prior_prec * x.pow(2)) - lamba, 1)
        logpw = torch.sum((1 - self.cdf_qz(0)) * logpw_col)
        return logpw
    
    def cdf_qz(self, x):
        limit_a, limit_b, epsilon = -.1, 1.1, 1e-6
        temperature = 2./3.
        xn = (x - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return F.sigmoid(logits * temperature - self.qz_loga).clamp(min=epsilon, max=1 - epsilon)



