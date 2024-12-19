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

#torch.set_printoptions(edgeitems=100)
torch.set_printoptions(precision=4)

def set_all_seeds(SEED):
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    
set_all_seeds(2023)


# In[ ]:


torch.cuda.get_device_name()


# In[3]:


from torch.utils.data import Dataset

def plot_loss(train_loss_arr, valid_loss_arr):
    
    fig, ax1 = plt.subplots(figsize=(20, 10))

    ax1.plot(train_loss_arr, 'k', label='NDCG by ApproxNDCG')
    ax1.plot(valid_loss_arr, 'g', label='NDCG by CE')
    ax1.legend(loc=1)
    ax2 = ax1.twinx()
    #ax2.plot(train_mape_arr, 'r--', label='train_mape_arr')
    #ax2.plot(v_mape_arr, 'b--', label='v_mape_arr')

    ax2.legend(loc=2)
    plt.show()
    plt.clf()
    
    return fig

def normalize(X):

    x_max = np.zeros(X.shape[3])
    x_min = np.zeros(X.shape[3])
    x_dif = np.zeros(X.shape[3])
    
    for i in range(X.shape[3]):
        x_max[i] = np.max(X[:,:,:,i])
        x_min[i] = np.min(X[:,:,:,i])

    x_dif = x_max - x_min
    
    x_dif = np.where(x_dif==0, 1, x_dif)
                
    new_X = np.zeros(X.shape)
        
    for ctr in range(X.shape[0]):
        for x in range(X.shape[1]):
            for y in range(X.shape[2]):
                new_X[ctr][x][y] = (X[ctr][x][y] - x_min)/x_dif
                    
    return new_X

def convert(Y_ori, mask):
    tmp = []
    for x in range(x_len):
        for y in range(y_len):

            if mask[x][y] != 1:
                continue

            try: 
                temp_x = mask[x-windows_size:x+windows_size+1,y-windows_size:y+windows_size+1]
            except:
                continue

            if temp_x.shape[0] != windows_size*2 + 1:
                continue
            if temp_x.shape[1] != windows_size*2 + 1:
                continue        

            temp_y = Y_ori[:,x,y]

            tmp.append(temp_y)

    tmp = np.array(tmp)
    tmp = tmp.transpose()

    Y_ori = tmp
    
    return Y_ori

def convert2(Y_ori, mask):
    tmp = []
    for x in range(x_len):
        for y in range(y_len):

            if mask[x][y] != 1:
                continue

            try: 
                temp_x = mask[x-windows_size:x+windows_size+1,y-windows_size:y+windows_size+1]
            except:
                continue

            if temp_x.shape[0] != windows_size*2 + 1:
                continue
            if temp_x.shape[1] != windows_size*2 + 1:
                continue        

            temp_y = Y_ori[x,y]

            tmp.append(temp_y)

    tmp = np.array(tmp)
    tmp = tmp.transpose()

    Y_ori = tmp
    
    return Y_ori


# In[ ]:


# chicago dataset
alpha_level = 0.05

X = np.load(open('NYC_X_train.npy', 'rb'))
Y = np.load(open('NYC_Y_train.npy', 'rb'))

Xt = np.load(open('NYC_X_test.npy', 'rb'))
Yt = np.load(open('NYC_Y_test.npy', 'rb'))

X = np.concatenate((X, Xt), axis=0)
Y = np.concatenate((Y, Yt), axis=0)

Y = Y[:, 0, :, :]

Y_ctr = Y.sum(axis=0)


mask = np.where(Y_ctr>100, 1, 0) # adjustable

print(mask.sum())

# X = normalize(X)

X_index = np.arange(len(X))

x_len, y_len = X.shape[2], X.shape[3]

Y_ori = Y.squeeze()

Y = np.where(Y >= 1, 1, 0) # binary classification.

#

# ----------------------------

print(X.shape)
print(Y.shape)

Xt = X[366:]
Xt_index = X_index[366:]
Yt = Y[366:]
X = X[:366]
X_index = X_index[:366]
Y = Y[:366]

temp_Y = Y

Xv = X[366-90:]
Xv_index = X_index[366-90:]
Yv = Y[366-90:]
X = X[:366-90]
X_index = X_index[:366-90]
Y = Y[:366-90]

Yv_verifiy = temp_Y[Xv_index]
Y_verifiy = temp_Y[X_index]

print("X, Y")
print(X.shape)
print(Y.shape)
print("Xv, Yv")
print(Xv.shape)
print(Yv.shape)
print("Xt, Yt")
print(Xt.shape)
print(Yt.shape)

del temp_Y

windows_size = 4

# In[ ]:


mask.sum()


# In[ ]:


print(Y.shape)

risk_map = np.zeros((x_len, y_len))

for x in range(x_len):
    for y in range(y_len):
        risk_map[x][y] = np.sum(Y[:, x, y])
        
fig, ax = plt.subplots(figsize=(5, 5))
ax = sns.heatmap(risk_map.transpose(), alpha=1, cbar=True, robust=False, annot=False)
ax.invert_yaxis()
plt.show()


# In[ ]:


import torch
import math
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair as pair
from torch.autograd import Variable
from torch.nn import init

def break_tensor_to_three_views(x):
    num_temporal = 15
    num_spatial = 18
    num_spatial_tempoal = 4
    temporal_view = x[:,:,0,0,0:num_temporal]
    spatial_view = x[:,0,:,:,num_temporal:num_temporal+num_spatial]
    spatial_tempoal_view = x[:,:,:,:,num_temporal+num_spatial:num_temporal+num_spatial+num_spatial_tempoal]
    return temporal_view, spatial_view, spatial_tempoal_view

def break_map_to_windows(X, Y):
    
    temporal_view, spatial_view, spatial_tempoal_view = break_tensor_to_three_views(X)
    del X
    
    new_spatial_tempoal_view, new_spatial_view, new_Y, coor_index = [], [], [], []
    
    for x in range(x_len):
        for y in range(y_len):
            
            #if risk_map[x][y] < 200:
            #    continue
                
            if mask[x][y] != 1:
                continue
            
            try:
                
                temp_x = spatial_tempoal_view[:,:,x-windows_size:x+windows_size+1,y-windows_size:y+windows_size+1,:]
                temp_spatial = spatial_view[:,x-windows_size:x+windows_size+1,y-windows_size:y+windows_size+1,:]
                temp_y = Y[:,x,y]
                temp_coor = [x, y]
                
            except:
                continue
            
            if temp_x.shape[2] != windows_size*2 + 1:
                continue
            if temp_x.shape[3] != windows_size*2 + 1:
                continue
                
            new_spatial_tempoal_view.append(temp_x)
            new_spatial_view.append(temp_spatial)
            new_Y.append(temp_y)
            coor_index.append(temp_coor)
            
    return np.array(new_spatial_tempoal_view), np.array(new_spatial_view), temporal_view, np.array(new_Y), np.array(coor_index)

def transpose_three_veiws(spatial_tempoal_view, spatial_view, temporal_view, Y):
    spatial_tempoal_view = np.transpose(spatial_tempoal_view, (1, 0, 2, 3, 4, 5))
    spatial_view = np.transpose(spatial_view, (1, 0, 2, 3, 4))
    Y = np.transpose(Y, (1, 0))
    return spatial_tempoal_view, spatial_view, temporal_view, Y

def convert_raw_matrix(X, Y):
    
    spatial_tempoal_view, spatial_view, temporal_view, Y, coor_index = break_map_to_windows(X, Y)
    spatial_tempoal_view, spatial_view, temporal_view, Y = transpose_three_veiws(spatial_tempoal_view, spatial_view, temporal_view, Y)
    temporal_view = temporal_view[:, 3, :]
    return spatial_tempoal_view, spatial_view, temporal_view, Y, coor_index

def pearson_corr(x, y):
    
    vx = x - torch.unsqueeze(torch.mean(x, dim = 1), 1)
    vy = y - torch.unsqueeze(torch.mean(y, dim = 1), 1)
    up = torch.mm(vx,  torch.t(vy))/y.shape[1]
    down = torch.sqrt(torch.mm(vx ** 2,  torch.t(vy ** 2))/y.shape[1])

    return 2*(up/down)

def exp_dist(x, y):

    spatial_tempoal_view = torch.cdist(x, y[None], p=2).squeeze(0)

    return torch.exp(-0.1*spatial_tempoal_view)


wide = int(windows_size*2 + 1)


import numpy as np
from scipy import signal
np.set_printoptions(suppress=True)

def gaussian_kernel(n, std, normalised=False):

    gaussian1D = signal.gaussian(n, std)
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    if normalised:
        gaussian2D /= (2*np.pi*(std**2))
    return gaussian2D

def direction_lookup(destination_x, origin_x, destination_y, origin_y):

    origin_x = origin_x - 1
    origin_y = origin_y - 1
    
    deltaX = destination_x - origin_x

    deltaY = destination_y - origin_y

    degrees_temp = math.atan2(deltaX, deltaY)/math.pi*180

    if degrees_temp < 0:

        degrees_final = 360 + degrees_temp

    else:

        degrees_final = degrees_temp

    # compass_brackets = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
    compass_brackets = ["N", "NW", "W", "SW", "S", "SE", "E", "NE", "N"] # special case in our problem
    
    compass_lookup = round(degrees_final / 45)

    return compass_brackets[compass_lookup], degrees_final

gussian_mask = {}

gussian_mask['M'] = gaussian_kernel(wide, 2, normalised=True)
gussian_mask['M'] = torch.tensor(gussian_mask['M'])
print(gussian_mask['M'].sum())

window_size = 3 # odd number
gap = int((window_size - 1)/2)

for x in range(gap, wide - gap):
    for y in range(gap, wide - gap):
        
        
        temp = np.zeros((wide, wide)).astype(int)

        for _x in range(window_size):
            for _y in range(window_size):

                temp[x-gap+_x][y-gap+_y] = 1
                
        gussian_mask[str(x) + '-' + str(y) + "_" + str(3)] = torch.tensor(temp)

coor_list = []
    
for x in range(gap, wide - gap):
    for y in range(gap, wide - gap):
        coor_list.append(str(x) + '-' + str(y) + "_" + str(3))

gussian_mask['ONE'] = gussian_mask['M']
gussian_mask['ONE'][:, :] = 1

window_size = 5 # odd number
gap = int((window_size - 1)/2)

for x in range(gap, wide - gap):
    for y in range(gap, wide - gap):
        
        
        temp = np.zeros((wide, wide)).astype(int)

        for _x in range(window_size):
            for _y in range(window_size):

                temp[x-gap+_x][y-gap+_y] = 1
                
        gussian_mask[str(x) + '-' + str(y) + "_" + str(5)] = torch.tensor(temp)
    
for x in range(gap, wide - gap):
    for y in range(gap, wide - gap):
        coor_list.append(str(x) + '-' + str(y) + "_" + str(5))
        


# In[ ]:


import numpy as np
from scipy import signal
np.set_printoptions(suppress=True)

def gaussian_kernel(n, std, normalised=False):

    gaussian1D = signal.gaussian(n, std)
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    if normalised:
        gaussian2D /= (2*np.pi*(std**2))
    return gaussian2D

def direction_lookup(destination_x, origin_x, destination_y, origin_y):

    origin_x = origin_x - 1
    origin_y = origin_y - 1
    
    deltaX = destination_x - origin_x

    deltaY = destination_y - origin_y

    degrees_temp = math.atan2(deltaX, deltaY)/math.pi*180

    if degrees_temp < 0:

        degrees_final = 360 + degrees_temp

    else:

        degrees_final = degrees_temp

    # compass_brackets = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
    compass_brackets = ["N", "NW", "W", "SW", "S", "SE", "E", "NE", "N"] # special case in our problem
    
    compass_lookup = round(degrees_final / 45)

    return compass_brackets[compass_lookup], degrees_final

gussian_mask2 = {}

all_directions_distance = ["N", "NW", "W", "SW", "S", "SE", "E", "NE", "Near", "Middle", "Far"]

for direcation in ["N", "NW", "W", "SW", "S", "SE", "E", "NE", "Near", "Middle", "Far"]:
    gussian_mask2[direcation] = torch.zeros(9, 9).to(device)

origin_x, origin_y = int((wide+1)/2), int((wide+1)/2)

temp = np.zeros((wide, wide)).astype(str)

for i in range(wide):
    for j in range(wide):
        
        temp[i][j] = direction_lookup(i, origin_x, j, origin_y)[0]
        
for i in range(wide):
    for j in range(wide):
        
        gussian_mask2[direction_lookup(i, origin_x, j, origin_y)[0]][i][j] = 1
        
print(gussian_mask2['SE'])
print(temp)

gussian_mask2['ONE'] = gussian_mask['M'].to(device)
gussian_mask2['ONE'][:, :] = 1


direcation_index = {}
for ctr, direcation in enumerate(["N", "NW", "W", "SW", "S", "SE", "E", "NE"]):
    direcation_index[direcation] = int(ctr)

for i in range(wide):
    for j in range(wide):
        
        temp[i][j] = direcation_index[temp[i][j]]
temp = temp.astype(int)
print(temp) 
fig, ax = plt.subplots(figsize=(5, 5))
ax = sns.heatmap(temp.transpose(), alpha=1, cbar=False, robust=False, annot=False, linewidths=0.5, linecolor='grey')
ax.invert_yaxis()
plt.show()


# In[ ]:


gussian_mask2['ONE']
origin_x, origin_y = origin_x-1, origin_y-1
print(origin_x, origin_y)
max_dist = 0

for i in range(wide):
    for j in range(wide):
        
        max_dist = max(math.sqrt((origin_x - i)**2 + (origin_y - j)**2), max_dist)
        
print(max_dist)

near_dist = max_dist/3
middle_dist = max_dist/3*2

print(near_dist, middle_dist)


# In[ ]:


temp = np.zeros((wide, wide)).astype(str)

for i in range(wide):
    for j in range(wide):
        
        dist = math.sqrt((origin_x - i)**2 + (origin_y - j)**2)
        
        if dist <= near_dist:
            gussian_mask2["Near"][i][j] = 1
            temp[i][j] = "Near"
        elif dist > middle_dist:
            gussian_mask2["Far"][i][j] = 1
            temp[i][j] = "Far"
        else:
            gussian_mask2["Middle"][i][j] = 1
            temp[i][j] = "Middle"
        
gussian_mask2["Middle"]


# In[ ]:


direcation_index = {}
for ctr, direcation in enumerate(["Near", "Middle", "Far"]):
    direcation_index[direcation] = int(ctr)

for i in range(wide):
    for j in range(wide):
        
        temp[i][j] = direcation_index[temp[i][j]]
        
temp = temp.astype(int)
print(temp) 
fig, ax = plt.subplots(figsize=(5, 5))
ax = sns.heatmap(temp.transpose(), alpha=1, cbar=False, robust=False, annot=False, linewidths=0.5, linecolor='grey')
ax.invert_yaxis()
plt.show()


# In[ ]:





# In[ ]:


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

def S_concept_encoding(S_view):
    print("S_encd...")

    b, loc, x, y, c = S_view.size()
    
    num_encd = 2 + 2
    
    concepts = torch.zeros((b*loc, c, len(coor_list), num_encd))
    # concepts = torch.zeros((b*loc, c, 1, num_encd))
    
    S_view_temp = S_view.clone().detach()
    
    S_view = S_view.flatten(start_dim=0, end_dim=1)
        
    for ctr, direcation in enumerate(coor_list):
        S_quantile_tmp = S_quantile[direcation]
        S_view2 = torch.mean(torch.mul(torch.tensor(S_view).permute(0,3,2,1), gussian_mask[direcation]).permute(0,3,2,1), (1, 2))*special_divider[direcation.split("_")[1]].clone().detach()
        S_view2 = S_view2.reshape(b, loc, c).cpu().detach().numpy()
        
        cc = 0
        for bat in range(b):
            
            for l in range(loc):
                if bat == 0:

                    pool_local = {}
                    for ch in range(c):

                        temp = S_view_temp[bat, l, :, :, ch].flatten().cpu().detach().numpy()
                        temp = temp[temp != 0] #

                        pool_local[ch] = temp
                
                    for ch in range(c):
                        pool = S_quantile_tmp[l][:, ch]
                        pool = pool[pool != 0] #
                        x = [S_view2[bat, l, ch]]

                        if len(pool) == 0 or x[0]==0:
                            continue
                        
                        ratio = stats.kstest(x, pool, alternative="two-sided")[1]
                        mu = np.mean(pool)

                        if ratio < alpha_level:
                            if x < mu:
                                concepts[cc, ch, ctr, 0] = 1
                            else:
                                concepts[cc, ch, ctr, 1] = 1
                            
                        if len(pool_local[ch]) == 0:
                            continue
                        
                        ratio = stats.kstest(x, pool_local[ch], alternative="two-sided")[1]
                        mu = np.mean(pool_local[ch])
                                                
                        if ratio < alpha_level:
                            if x < mu:
                                concepts[cc, ch, ctr, 2] = 1
                            else:
                                concepts[cc, ch, ctr, 3] = 1
                            
                else:
                    
                    concepts[cc, ch, ctr, 0] = concepts[l, ch, ctr, 0]
                    concepts[cc, ch, ctr, 1] = concepts[l, ch, ctr, 1]
                    concepts[cc, ch, ctr, 2] = concepts[l, ch, ctr, 2]
                    concepts[cc, ch, ctr, 3] = concepts[l, ch, ctr, 3]
                cc += 1 
                
        # ------------------------------------------------------------------------------------------------------
    
    concepts = concepts.reshape(b, loc, -1)
    print(concepts.shape)
    return concepts
    
def T_concept_encoding(T_view):
    print("T_encd...")
    # b, 12
    # [2000, 48]
    
    b, f = T_view.size()
    # T_view = torch.mean(T_view, 1)
    out = torch.zeros((b, f, 2))
    for i in range(b):
        for j in range(f):
            
            pool = T_quantile[:, j]
            pool = pool[pool != 0] #
            
            if len(pool) == 0:
                continue
                
            x = [T_view[i, j].cpu().detach().numpy()]
            
            ratio = stats.kstest(x, pool, alternative="two-sided")[1]
            mu = pool.mean()
            
            if ratio < alpha_level:
                if x < mu:
                    out[i, j, 0] = 1
                else:
                    out[i, j, 1] = 1       
    
    out = out.flatten(start_dim=1, end_dim=2)
    print(out.shape)
    return out

def ST_concept_encoding(ST_view):
    print("ST_encd...")

    b, loc, seq, x, y, c = ST_view.size()
    
    num_encd = 2 + 2
    
    concepts = torch.zeros((b*loc, c, len(coor_list), num_encd))
    # concepts = torch.zeros((b*loc, c, 1, num_encd))

    
    ST_view_temp = ST_view.clone().detach()
    print(ST_view_temp.shape)
    
    ST_view = ST_view.flatten(start_dim=0, end_dim=1)
    
    
    for ctr, direcation in enumerate(coor_list):
        SP_quantile_tmp = SP_quantile[direcation]
        # SP_quantile_tmp2 = SP_quantile_region[direcation].cpu().detach().numpy()

        # ST_view2 = torch.sum(torch.mul(ST_view.permute(0,1,4,3,2), gussian_mask[direcation]).permute(0,1,4,3,2), (1, 2, 3))/(1/81)
        ST_view2 = torch.mean(torch.mul(torch.tensor(ST_view).permute(0,1,4,3,2), gussian_mask[direcation]).permute(0,1,4,3,2), (1, 2, 3))*special_divider[direcation.split("_")[1]].clone().detach()
        ST_view2 = ST_view2.reshape(b, loc, c).numpy()

        cc = 0
        for bat in range(b):
            
            for l in range(loc):
                
                pool_local = {}

                for ch in range(c):
                    temp = ST_view_temp[bat, l, :, :, :, ch].flatten().cpu().detach().numpy()
                    temp = temp[temp != 0] #
                    pool_local[ch] = temp
                
                for ch in range(c):
                    
                    pool = SP_quantile_tmp[l][:, ch]
                    pool = pool[pool != 0] #
                    x = ST_view2[bat, l, ch]

                    if len(pool) == 0 or x==0:
                        continue
                                        
                    mu = np.mean(pool)
                    
                    ratio_less = poisson.cdf(x, mu)
                    ratio_more = 1 - poisson.cdf(x, mu)

                    if ratio_less < alpha_level:
                        concepts[cc, ch, ctr, 0] = 1
                    elif ratio_more < alpha_level:
                        concepts[cc, ch, ctr, 1] = 1
                        
                    if len(pool_local[ch]) == 0:
                        continue
                    
                    mu = pool_local[ch].mean()
                    
                    ratio_less = poisson.cdf(x, mu)
                    ratio_more = 1 - poisson.cdf(x, mu)
                    
                    if ratio_less < alpha_level:
                        concepts[cc, ch, ctr, 2] = 1
                    elif ratio_more < alpha_level:
                        concepts[cc, ch, ctr, 3] = 1
                             
                cc += 1
                
        # ------------------------------------------------------------------------------------------------------
        
    concepts = concepts.reshape(b, loc, -1)
    print(concepts.shape)

    return concepts

def max_directional_pooling(target):
    ppd = nn.ZeroPad2d(1)
    b_loc, ch, x, y = target.shape
    concepts = torch.zeros((b_loc, len(all_directions_distance), ch)).to(device)
    
    for ctr, each in enumerate(all_directions_distance):
        temp = torch.mul(ppd(target), gussian_mask2[each])
        # torch.Size([32384, 24, 9, 9]) -> (9, 6, 4)
        temp = torch.sum(temp, (2, 3))
        temp = torch.where(temp > 0, 1, 0)
        concepts[:, ctr] = temp
            
    return concepts

def avg_directional_pooling(target):
    ppd = nn.ZeroPad2d(1)
    b_loc, ch, x, y = target.shape
    concepts = torch.zeros((b_loc, len(all_directions_distance), ch)).to(device)
    
    for ctr, each in enumerate(all_directions_distance):
        temp = torch.mul(ppd(target), gussian_mask2[each])
        # torch.Size([32384, 24, 9, 9]) -> (9, 6, 4)
        temp = torch.sum(temp, (2, 3))/torch.sum(gussian_mask2[each])
        concepts[:, ctr] = temp
            
    return concepts

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
                
        spatial_proto = 1/torch.cdist(output, self.S_prototype_vectors[None], p=2).squeeze(0)
        
        shad = spatial_proto.clone()
        shad -= shad.min(1, keepdim=True)[0]
        shad /= shad.max(1, keepdim=True)[0]
        spatial_proto = shad
        
        output = self.FC(spatial_proto)
        output = self.soft(output)

        loss_l0 = self.L0(self.S_prototype_vectors)
        res = torch.ones(860).to(device)

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


# In[ ]:


torch.sum(gussian_mask2["N"])


# In[ ]:


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


# In[ ]:


import datetime
now = datetime.datetime.now()
print(now)


# In[ ]:



batch_size = 64

spatial_tempoal_view, spatial_view, temporal_view, Y2, coor_index = convert_raw_matrix(X, Y)
q = torch.tensor([0.25, 0.5, 0.75], dtype = torch.float64)

special_divider = {}
special_divider["3"] = torch.sum(gussian_mask['ONE'])/9
special_divider["5"] = torch.sum(gussian_mask['ONE'])/25

# T_quantile = torch.quantile(torch.tensor(temporal_view), q, dim=0, keepdim=False)
# computing q1 q2 q3 q4
print(temporal_view.shape)

with open('T_quantile_NYC.pkl', 'wb') as f:
    pickle.dump(temporal_view, f)
    

S_quantile = {}
S_quantile_local = {}

print(spatial_view.shape)
pool = []

for direcation in coor_list:

    S_quantile[direcation] = {}
    S_quantile_local[direcation] = {}
    
    for loc in range(len(coor_index)):

        loc_sp = torch.tensor(spatial_view[:, loc])
        loc_xy = torch.tensor(coor_index[loc])
        # loc_sp = torch.sum(torch.mul(torch.tensor(loc_sp).permute(0,1,4,3,2), gussian_mask[direcation]).permute(0,1,4,3,2), (1, 2, 3))/(1/torch.sum(gussian_mask[direcation]))
        loc_sp = torch.mean(torch.mul(torch.tensor(loc_sp).permute(0,3,2,1), gussian_mask['ONE']).permute(0,3,2,1), (0, 1, 2)).clone().detach()
        # loc_sp = torch.quantile(loc_sp, q, dim=0, keepdim=False)
        # loc_sp = loc_sp.transpose(1, 0)
        pool.append(loc_sp[None, :])
        
pool = torch.cat(pool)
pool = pool.cpu().detach().numpy()

for direcation in coor_list:

    for loc in range(len(coor_index)):
        S_quantile[direcation][loc] = pool

print("done")
    
with open('S_quantile_NYC.pkl', 'wb') as f:
    pickle.dump(S_quantile, f)

SP_quantile = {}

pool = []

print(spatial_tempoal_view.shape)
for direcation in coor_list:

    SP_quantile[direcation] = {}
    

    for loc in range(len(coor_index)):

        loc_aa = torch.tensor(spatial_tempoal_view[:, loc])
        loc_xy = torch.tensor(coor_index[loc])
        loc_sp = torch.mean(torch.mul(torch.tensor(loc_aa).permute(0,1,4,3,2), gussian_mask[direcation]).permute(0,1,4,3,2), (0, 1, 2, 3))*special_divider[direcation.split("_")[1]].clone().detach()
        pool.append(loc_sp[None, :])
            
pool = torch.cat(pool)
pool = pool.cpu().detach().numpy()

for direcation in coor_list:
    for loc in range(len(coor_index)):
        SP_quantile[direcation][loc] = pool
    

with open('SP_quantile_NYC.pkl', 'wb') as f: # _one_NYC
    pickle.dump(SP_quantile, f)


with open('T_quantile_NYC.pkl', 'rb') as f: # 
    T_quantile = pickle.load(f)
    
with open('SP_quantile_NYC.pkl', 'rb') as f: # 
    SP_quantile = pickle.load(f)


with open('S_quantile_NYC.pkl', 'rb') as f: # 
    S_quantile = pickle.load(f)


train_dataset = DM_Dataset(spatial_tempoal_view, spatial_view, temporal_view, Y2, coor_index, X_index)
training_generator = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

with open('training_generator_NYC.pkl', 'wb') as f:
    pickle.dump(training_generator, f)
    
print("training done")

spatial_tempoal_view, spatial_view, temporal_view, Yv2, coor_index = convert_raw_matrix(Xv, Yv)
validation_dataset = DM_Dataset(spatial_tempoal_view, spatial_view, temporal_view, Yv2, coor_index, Xv_index)
validation_generator = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

with open('validation_generator_NYC.pkl', 'wb') as f:
    pickle.dump(validation_generator, f)
    
print("vlidating done")

spatial_tempoal_view, spatial_view, temporal_view, Yt2, coor_index = convert_raw_matrix(Xt, Yt)
test_dataset = DM_Dataset(spatial_tempoal_view, spatial_view, temporal_view, Yt2, coor_index, Xt_index)
test_generator = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

with open('test_generator_NYC.pkl', 'wb') as f:
    pickle.dump(test_generator, f)
    
print("testing done")

# ------------------------------------------------ comment everything above to avoid re-encoding

with open('training_generator_NYC.pkl', 'rb') as f: #
    training_generator = pickle.load(f)

with open('validation_generator_NYC.pkl', 'rb') as f: # 
    validation_generator = pickle.load(f)
    
with open('test_generator_NYC.pkl', 'rb') as f: # 
    test_generator = pickle.load(f)


# In[ ]:





# In[ ]:


import datetime
now = datetime.datetime.now()
print(now)


# In[ ]:


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


# In[ ]:


def y_softmax_modifier(y):
    
    y = y.flatten()
    
    new_y = torch.zeros(len(y), 2).to(device)
    
    new_y[:, 0] = torch.where(y==0, 1, 0).clone()
    new_y[:, 1] = y.clone()
    
    return new_y

CE = nn.CrossEntropyLoss()

model = ProtoNet().to(device) # chicago

#model.load_state_dict(torch.load('model.pth'))
learning_rate = 1e-3
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)

train_loss_arr = []
valid_loss_arr =[]
acc_loss_arr = []
recall_loss_arr = []

best_result = math.inf
best_ctr = 0

for echo in range(50):
    print("echo: " + str(echo))
    avg_train_loss = []
    avg_valid_loss = []
    avg_acc_loss = []
    avg_recall_loss = []
    
    model.train()

    for (spatial_tempoal_view, ST_encd, T_encd, S_encd), spatial_view, temporal_view, l_Y, l_coor, l_X_index in training_generator:
        spatial_tempoal_view, ST_encd, spatial_view, temporal_view = spatial_tempoal_view.to(device), ST_encd.to(device), spatial_view.to(device), temporal_view.to(device)
        l_Y, l_coor, T_encd, S_encd = l_Y.to(device), l_coor.to(device), T_encd.to(device), S_encd.to(device)
        
        
        # tmp_y = l_Y.clone()
        l_Y = y_softmax_modifier(l_Y)

        # print(ST_encd.shape)
        outputs, _, prototype, _, loss_l0 = model(ST_encd, S_encd, T_encd)
        # loss_diver = diversity_cost_spatial2(model.S_prototype_vectors)
    
        # l_Y = l_Y.flatten(0)
        # outputs = outputs.flatten(0)
        
        train_loss = CE(outputs, l_Y)
        _, loss_sep, loss_clu, _ = criterion(outputs, l_Y, prototype)
        # loss_diver_S = 80*diversity_cost_spatial("S_prototype_vectors")
        
        #print('-------------------------------')
        #print(train_loss)
        #print(80*loss_diver_S)
        
        # train_loss = (loss_CE + loss_sep + loss_clu + loss_diver + loss_diver_S)
        # train_loss = loss_CE + loss_diver + loss_diver_S
        train_loss = train_loss + 1*loss_l0 + 0.01*loss_sep + 0.01*loss_clu # + 0.1*loss_l0.flatten(0)[0] + 1000*loss_diver # + loss_sep + loss_clu
                
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        avg_train_loss.append(train_loss.cpu().data)
        
    train_loss_arr.append(np.average(avg_train_loss))

    #print(train_loss)
    #print(ndcg_score(local_labels.cpu().detach().numpy(), outputs.cpu().detach().numpy()))
    #print(np.average(avg_train_loss))

    model.eval()

    with torch.no_grad():

        for (spatial_tempoal_view, ST_encd, T_encd, S_encd), spatial_view, temporal_view, l_Y, l_coor, l_X_index in validation_generator:
            spatial_tempoal_view, ST_encd, spatial_view, temporal_view = spatial_tempoal_view.to(device), ST_encd.to(device), spatial_view.to(device), temporal_view.to(device)
            
            b_size, loc_size = l_Y.shape
            
            l_Y, l_coor, T_encd, S_encd = l_Y.to(device), l_coor.to(device), T_encd.to(device), S_encd.to(device)
            l_Y = y_softmax_modifier(l_Y)

            Voutputs, _, _, _, loss_diver_S = model(ST_encd, S_encd, T_encd)
            
            # l_Y = l_Y.flatten(0)
            # Voutputs = Voutputs.flatten(0)
        
            v_loss = CE(Voutputs, l_Y)
            acc = accuracy_torch(Voutputs, l_Y)
            
            recall = recall_torch(Voutputs, l_Y)

            avg_valid_loss.append(v_loss.cpu().data)
            avg_acc_loss.append(acc.cpu().data)
            avg_recall_loss.append(recall)

    valid_loss_arr.append(np.average(avg_valid_loss))
    acc_loss_arr.append(np.average(avg_acc_loss))
    recall_loss_arr.append(np.average(avg_recall_loss))

    if best_result <= float(valid_loss_arr[-1].item()):
        best_ctr += 1
    else:
        best_ctr = 0
        print("epochs: " + str(echo))
        
        print(float(valid_loss_arr[-1].item()))

        with open("model.pkl",'wb') as f:
            pickle.dump(model,f)
            
        print("model saved")
        print('----------------------------------------------------------------------------')

    best_result = min(best_result, valid_loss_arr[-1].item())
    
    
    if best_ctr > 5:
        print("early stop")
        break
    # print(loss_CE, loss_sep, loss_clu, loss_diver, loss_diver_S)


# In[ ]:


post = "proto"

with open("model" + post + ".pkl",'wb') as f:
    pickle.dump(model,f)
            
for name, parameter in model.named_parameters():
    print(name, parameter.size())
    
model.FC.weight

np.save("FC_weights"+post+".npy", model.FC.weight.cpu().detach().numpy())
np.save("loc_weights"+post+".npy", model.loc_weights.cpu().detach().numpy())


# In[ ]:


torch.set_printoptions(threshold=100000, precision=2, sci_mode=False)
# model.S_prototype_vectors[3]

prototype_vector = model.S_prototype_vectors.clone()
prototype_vector -= prototype_vector.min(1, keepdim=True)[0]
prototype_vector /= prototype_vector.max(1, keepdim=True)[0]
np.save("res"+post+".npy", prototype_vector.cpu().detach().numpy())

prototype_vector.shape


# In[ ]:


all_pred = []
all_true = []

all_pred2 = []
all_true2 = []

with torch.no_grad():

    for (spatial_tempoal_view, ST_encd, T_encd, S_encd), spatial_view, temporal_view, l_Y, l_coor, l_X_index in test_generator:
        spatial_tempoal_view, ST_encd, spatial_view, temporal_view = spatial_tempoal_view.to(device), ST_encd.to(device), spatial_view.to(device), temporal_view.to(device)
        l_Y, l_coor, T_encd, S_encd = l_Y.to(device), l_coor.to(device), T_encd.to(device), S_encd.to(device)
        b_size, loc_size = l_Y.shape

        l_Y = y_softmax_modifier(l_Y)

        Voutputs, _, _, _, loss_diver_S = model(ST_encd, S_encd, T_encd)

        # l_Y = l_Y.flatten(0)
        # Voutputs = Voutputs.flatten(0)

        all_pred.append(Voutputs)
        all_true.append(l_Y)

        all_pred2.append(Voutputs)
        all_true2.append(l_Y)
        
all_pred = torch.cat(all_pred)
all_true = torch.cat(all_true)
all_pred2 = torch.cat(all_pred2)
all_true2 = torch.cat(all_true2)


# In[ ]:


thresh = 0.43
v_loss = CE(all_pred, all_true)
acc = accuracy_torch(all_pred, all_true, thresh=thresh)
recall = recall_torch(all_pred2, all_true2, thresh=thresh)
f1 = f1_score_torch(all_pred2, all_true2, thresh=thresh)
precision = precision_score_torch(all_pred2, all_true2, thresh=thresh)
print(acc, precision, recall, f1)
print(v_loss)



# In[ ]:



model.eval()

true_y_saver, pred_y_saver, path_prob_saver = [], [], []

with torch.no_grad():
    ctr = 0
    for (spatial_tempoal_view, ST_encd, T_encd, S_encd), spatial_view, temporal_view, l_Y, l_coor, l_X_index in training_generator:
        spatial_tempoal_view, ST_encd, spatial_view, temporal_view = spatial_tempoal_view.to(device), ST_encd.to(device), spatial_view.to(device), temporal_view.to(device)
        l_coor = l_coor.to(device)
        T_encd = T_encd.to(device)
        S_encd = S_encd.to(device)
        pred_y, _, path_prob, _, _ = model(ST_encd, S_encd, T_encd)
        
        if ctr == 0:
            pred_y_saver = pred_y.flatten(0).cpu().detach().numpy()
            true_y_saver = l_Y.flatten(0)
            path_prob_saver = path_prob.cpu().detach().numpy()

        else:
            pred_y_saver = np.concatenate((pred_y_saver, pred_y.flatten(0).cpu().detach().numpy()), axis=0)
            true_y_saver = np.concatenate((true_y_saver, l_Y.flatten(0)), axis=0)
            path_prob_saver = np.concatenate((path_prob_saver, path_prob.cpu().detach().numpy()), axis=0)

        ctr += 1

np.save("pred_y_saver"+post+".npy", pred_y_saver)
np.save("true_y_saver"+post+".npy", true_y_saver)
np.save("path_prob_saver"+post+".npy", path_prob_saver)

print("projection set saved")


# In[ ]:





# In[ ]:


model.eval()

spatial_tempoal_saver, temporal_saver, spatial_saver = [], [], []
ST, T, S, res_arr = [], [], [], []
ST_encoded_arr, T_encoded_arr, S_encoded_arr, S_encd = [], [], [], []
coor_index_arr = []
index_box = []

with torch.no_grad():
    ctr = 0
    for (spatial_tempoal_view, ST_encd, T_encd, S_encd), spatial_view, temporal_view, l_Y, l_coor, l_X_index in training_generator:
        spatial_tempoal_view, ST_encd, spatial_view, temporal_view = spatial_tempoal_view.to(device), ST_encd.to(device), spatial_view.to(device), temporal_view.to(device)
        l_Y, l_coor, T_encd, S_encd = l_Y.to(device), l_coor.to(device), T_encd.to(device), S_encd.to(device)
        pred_y, _, spatial_proto, (ST_encd, T_encd, S_encd, res),_ = model(ST_encd, S_encd, T_encd)
        if ctr == 0:
            spatial_tempoal_saver = spatial_tempoal_view.cpu().detach().numpy()
            #temporal_saver = temporal_view.cpu().detach().numpy()
            spatial_saver = spatial_view.cpu().detach().numpy()
            
            ST_encoded_arr = ST_encd.cpu().detach().numpy()
            T_encoded_arr = T_encd.cpu().detach().numpy()
            S_encoded_arr = S_encd.cpu().detach().numpy()
            coor_index_arr = l_coor.cpu().detach().numpy()
            
            #T = temporal_proto.cpu().detach().numpy()
            S = spatial_proto.cpu().detach().numpy()
            res_arr = res.cpu().detach().numpy()
            
            index_box = l_X_index

        else:
            spatial_tempoal_saver = np.concatenate((spatial_tempoal_saver, spatial_tempoal_view.cpu().detach().numpy()), axis=0)
            #temporal_saver = np.concatenate((temporal_saver, temporal_view.cpu().detach().numpy()), axis=0)
            spatial_saver = np.concatenate((spatial_saver, spatial_view.cpu().detach().numpy()), axis=0)
            ST_encoded_arr = np.concatenate((ST_encoded_arr, ST_encd.cpu().detach().numpy()), axis=0)
            T_encoded_arr = np.concatenate((T_encoded_arr, T_encd.cpu().detach().numpy()), axis=0)
            S_encoded_arr = np.concatenate((S_encoded_arr, S_encd.cpu().detach().numpy()), axis=0)
            #T = np.concatenate((T, temporal_proto.cpu().detach().numpy()), axis=0)
            S = np.concatenate((S, spatial_proto.cpu().detach().numpy()), axis=0)
            coor_index_arr = np.concatenate((coor_index_arr, l_coor.cpu().detach().numpy()), axis=0)

            index_box = np.concatenate((index_box, l_X_index), axis=0)

        ctr += 1

np.save("spatial_tempoal_saver"+post+".npy", spatial_tempoal_saver)
#np.save("D:/PrototypeLearning/saved_proto_valid/temporal_saver.npy", temporal_saver)

np.save("spatial_saver"+post+".npy", spatial_saver)

np.save("ST_encd"+post+".npy", ST_encoded_arr)
np.save("T_encd"+post+".npy", T_encoded_arr)
np.save("S_encd"+post+".npy", S_encoded_arr)

#np.save("D:/PrototypeLearning/saved_proto_valid/T.npy", T)
np.save("S"+post+".npy", S)
np.save("res"+post+".npy", prototype_vector.cpu().detach().numpy())

np.save("index_box"+post+".npy", index_box)
np.save("coor_index"+post+".npy", coor_index_arr)
print(coor_index_arr.shape)
print("training projection saved")


# In[ ]:




