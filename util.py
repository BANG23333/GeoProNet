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



def diversity_cost_spatial(proto_key):

    prototype_vector = model.state_dict()[proto_key]
    loss_diver = 0
    ctr = 0
    for i in range(prototype_vector.size()[0]):
        for j in range(i+1, prototype_vector.size()[0]):
            ctr += 1
            loss_diver += torch.cdist(prototype_vector[i].expand(1, -1), prototype_vector[j].expand(1, -1))
    
    loss_diver = -1 * loss_diver/ctr

    return loss_diver

def diversity_cost_spatial2(protos):

    prototype_vector = protos.clone()
    prototype_vector -= prototype_vector.min(1, keepdim=True)[0]
    prototype_vector /= prototype_vector.max(1, keepdim=True)[0]
    
    loss_diver = 0
    ctr = 0
    
    for i in range(protos.size()[0]):
        for j in range(protos.size()[0]):
            if i == j:
                continue
            ctr += 1
            loss_diver += torch.cdist(prototype_vector[i].expand(1, -1), prototype_vector[j].expand(1, -1), p=2)
    
    loss_diver = -1 * loss_diver/ctr

    return loss_diver

def criterion(outputs, l_Y, spatial_tempoal_proto):

    l_Y = l_Y[:, 0].flatten(0)
    prototypes_of_positive_class = torch.where(model.state_dict()["FC.weight"][0][:, None] >= 0, 1, 0).permute(1, 0)
    prototypes_of_negative_class = 1 - prototypes_of_positive_class
    positive_correct = spatial_tempoal_proto*prototypes_of_positive_class
    positive_correct = torch.t(l_Y*torch.t(positive_correct))
    positive_wrong = spatial_tempoal_proto*prototypes_of_negative_class
    positive_wrong = torch.t(l_Y*torch.t(positive_wrong))
    
    negative_correct = spatial_tempoal_proto*prototypes_of_negative_class
    negative_correct = torch.t((1-l_Y)*torch.t(negative_correct))
    negative_wrong = spatial_tempoal_proto*prototypes_of_positive_class
    negative_wrong = torch.t((1-l_Y)*torch.t(negative_wrong))
    
    #separation cost
    loss_sep = -1*(torch.sum(positive_wrong)/torch.count_nonzero(positive_wrong)+torch.sum(negative_wrong)/torch.count_nonzero(negative_wrong))
    
    #clustering cost
    loss_clu = torch.sum(positive_correct)/torch.count_nonzero(positive_correct)+torch.sum(negative_correct)/torch.count_nonzero(negative_correct)
    
    # diveristy cost
        
    return 0, loss_sep.clone().detach(), loss_clu.clone().detach(), 0


from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score

def accuracy_torch(vector_x, vector_y, thresh = 0.5):

    vector_x = torch.where(vector_x > thresh, 1, 0).flatten(0)
    vector_y = torch.where(vector_y > thresh, 1, 0).flatten(0)

    return torch.sum(vector_x==vector_y)/len(vector_x)

def recall_torch(vector_x, vector_y, thresh = 0.5):
    
    vector_x = torch.where(vector_x > thresh, 1, 0)
    vector_y = torch.where(vector_y > thresh, 1, 0)
    
    vector_x = vector_x[:, 1].cpu().detach().numpy()
    vector_y = vector_y[:, 1].cpu().detach().numpy()
    
    return recall_score(vector_y, vector_x)

def f1_score_torch(vector_x, vector_y, thresh = 0.5):
    
    vector_x = torch.where(vector_x > thresh, 1, 0)
    vector_y = torch.where(vector_y > thresh, 1, 0)
    
    vector_x = vector_x[:, 1].cpu().detach().numpy()
    vector_y = vector_y[:, 1].cpu().detach().numpy()
    
    return f1_score(vector_y, vector_x)

def precision_score_torch(vector_x, vector_y, thresh = 0.5):
    
    vector_x = torch.where(vector_x > thresh, 1, 0)
    vector_y = torch.where(vector_y > thresh, 1, 0)
    
    vector_x = vector_x[:, 1].cpu().detach().numpy()
    vector_y = vector_y[:, 1].cpu().detach().numpy()
    
    return precision_score(vector_y, vector_x)