#!/usr/bin/env python
# coding: utf-8

# # Attribute
# 
# **Original Work**: *Maziar Raissi, Paris Perdikaris, and George Em Karniadakis*
# 
# **Github Repo** : https://github.com/maziarraissi/PINNs
# 
# **Link:** https://github.com/maziarraissi/PINNs/tree/master/appendix/continuous_time_identification%20(Burgers)
# 
# @article{raissi2017physicsI,
#   title={Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations},
#   author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em},
#   journal={arXiv preprint arXiv:1711.10561},
#   year={2017}
# }
# 
# @article{raissi2017physicsII,
#   title={Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear Partial Differential Equations},
#   author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em},
#   journal={arXiv preprint arXiv:1711.10566},
#   year={2017}
# }

# ## Libraries and Dependencies

# In[1]:


import sys
sys.path.insert(0, '../Utilities/')

import torch
from collections import OrderedDict

from pyDOE import lhs
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import time

np.random.seed(1234)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class collocationNet(torch.nn.Module):
    def __init__(self, layers=None):
        super(collocationNet, self).__init__()

        self.fc1 = torch.nn.Linear(9,20)
        self.fc2 = torch.nn.Linear(20,20)
        self.fc3 = torch.nn.Linear(20,20)
        self.fc4 = torch.nn.Linear(20,20)
        self.fc5 = torch.nn.Linear(20,20)
        self.fc6 = torch.nn.Linear(20,20)
        self.fc7 = torch.nn.Linear(20,20)
        self.fc8 = torch.nn.Linear(20,100)
        # parameters
        # self.depth = len(layers) - 1
        
        # # set up layer order dict
        self.activation = torch.nn.Tanh()

        # if layers is None:
        #     layers = [9, 20, 20, 20, 100]

        # # parameters
        # self.depth = len(layers) - 1
        # print("depth ",self.depth)
        # # set up layer order dict
        # # self.activation = torch.nn.Sigmoid
        
        # layer_list = list()
        # for i in range(self.depth - 1): 
        #     layer_list.append(
        #         ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
        #     )
        #     # layer_list.append(('activation_%d' % i, self.activation()))
            
        # layer_list.append(
        #     ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        # )
        # layerDict = OrderedDict(layer_list)
        
        # # deploy layers
        # self.layers = torch.nn.Sequential(layerDict)
        
    def forward(self, pv):
        pv = pv.float()
        out = self.activation(self.fc1(pv))
        out = self.activation(self.fc2(out))
        out = self.activation(self.fc3(out))
        out = self.activation(self.fc4(out))
        out = self.activation(self.fc5(out))
        out = self.activation(self.fc6(out))
        out = self.activation(self.fc7(out))
        out = self.fc8(out)
        # output = torch.zeros(100,2)
        # raw_c_points = self.layers(pv.float())
        # print("raw c points shape:",raw_c_points.shape)
        # out[:50,0]=raw_c_points[0,:50]
        # out[50:,1]=raw_c_points[0,50:]
        # out[50:75,0]=1
        # out[75:,0]=-1
        # print("cn")
        
        # x = out[:,0].unsqueeze(1)
        # t = out[:,1].unsqueeze(1)

        return out


# In[10]:


# the deep neural network
class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
        
        self.fc1 = torch.nn.Linear(2,20)
        self.fc2 = torch.nn.Linear(20,20)
        self.fc3 = torch.nn.Linear(20,20)
        self.fc4 = torch.nn.Linear(20,20)
        self.fc5 = torch.nn.Linear(20,20)
        self.fc6 = torch.nn.Linear(20,20)
        self.fc7 = torch.nn.Linear(20,20)
        self.fc8 = torch.nn.Linear(20,1)
        # parameters
        # self.depth = len(layers) - 1
        
        # # set up layer order dict
        self.activation = torch.nn.Tanh()
        
        # layer_list = list()
        # for i in range(self.depth - 1): 
        #     layer_list.append(
        #         ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
        #     )
        #     layer_list.append(('activation_%d' % i, self.activation()))
            
        # layer_list.append(
        #     ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        # )
        # layerDict = OrderedDict(layer_list)
        
        # # deploy layers
        # self.layers = torch.nn.Sequential(layerDict)
        

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(self.fc2(out))
        out = self.activation(self.fc3(out))
        out = self.activation(self.fc4(out))
        out = self.activation(self.fc5(out))
        out = self.activation(self.fc6(out))
        out = self.activation(self.fc7(out))
        out = self.fc8(out)
        return out


# In[11]:


# the physics-guided neural network
class PhysicsInformedNN():
    def __init__(self, X_u, u, X_f, layers, lb, ub, nu,pv):
        
        self.pv = torch.tensor(pv).float().to(device)

        # boundary conditions
        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)
        
        # data
        self.x_u = torch.tensor(X_u[:, 0:1], requires_grad=True).float().to(device) #BC x coordiantes
        self.t_u = torch.tensor(X_u[:, 1:2], requires_grad=True).float().to(device) #BC y coordinates
        self.x_f = torch.tensor(X_f[:, 0:1], requires_grad=True).float().to(device) #test sample points in x?
        self.t_f = torch.tensor(X_f[:, 1:2], requires_grad=True).float().to(device) #test sample points in t?
        self.u = torch.tensor(u).float().to(device) #IC and BC u values
        
        self.output = torch.zeros(100,2)
        self.raw_c_points = torch.zeros(1,100)
        self.layers = layers
        self.nu = nu
        
        # deep neural networks
        self.cnet = collocationNet().to(device)
        self.dnn = DNN(layers).to(device)
        
        # optimizers: using the same settings
        self.optimizer = torch.optim.LBFGS(
            list(self.dnn.parameters())+list(self.cnet.parameters()), 
            lr=1.0, 
            max_iter=50000, 
            max_eval=50000, 
            history_size=50,
            tolerance_grad=1e-5, 
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"       # can be "strong_wolfe"
        )

        self.iter = 0
        
    def net_u(self, pv):
        self.raw_c_points = self.cnet(torch.tensor(pv.float()).double().to(device)) #[x,t] = self.cnet(torch.tensor(pv))
        self.output[:50,0]=self.raw_c_points[0,:50]
        self.output[50:,1]=self.raw_c_points[0,50:]
        self.output[50:75,0]=1
        self.output[75:,0]=-1
        x = self.output[:,0].unsqueeze(1).to(device)
        t = self.output[:,1].unsqueeze(1).to(device)
        u = self.dnn(torch.cat([x, t], dim=1)) #u = self.dnn(torch.cat([x, t], dim=1))
        return u
    
    def val_net_u(self, x, t):  
        u = self.dnn(torch.cat([x, t], dim=1))
        return u

    def net_f(self, x, t):
        """ The pytorch autograd version of calculating residual """
        pv = self.pv
        u = self.val_net_u(x,t)
        
        u_t = torch.autograd.grad(
            u, t, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True,
            allow_unused=True
        )[0]
        u_x = torch.autograd.grad(
            u, x, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True,allow_unused=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x, 
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True,allow_unused=True
        )[0]
        u_tt = torch.autograd.grad(
            u_t, t, 
            grad_outputs=torch.ones_like(u_t),
            retain_graph=True,allow_unused=True,
            create_graph=True
        )[0]
        u_xt = torch.autograd.grad(
            u_x, t, 
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,allow_unused=True,
            create_graph=True
        )[0]
        f = pv[:,0]*u+pv[:,1]*u_t + pv[:,2]*u_x + pv[:,3]*u_tt + pv[:,4]*u_xx
        return f
    
    def loss_func(self):
        self.optimizer.zero_grad()
        
        #now we deal with this:
        u_pred = self.net_u(self.pv)#(self.x_u, self.t_u)
        print("check here")
        f_pred = self.net_f(self.x_f, self.t_f)
        loss_u = torch.mean((self.u - u_pred) ** 2)
        loss_f = torch.mean(f_pred ** 2)
        
        loss = loss_u + loss_f
        torch.autograd.set_detect_anomaly(True)
        loss.backward(retain_graph=True)
        self.iter += 1
        if self.iter % 100 == 0:
            print(
                'Iter %d, Loss: %.5e, Loss_u: %.5e, Loss_f: %.5e' % (self.iter, loss.item(), loss_u.item(), loss_f.item())
            )
        return loss
    
    def train(self):
        self.cnet.train()
        self.dnn.train()
                
        # Backward and optimize
        self.optimizer.step(self.loss_func)

            
    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
        self.cnet.eval()
        self.dnn.eval()
        u = self.val_net_u(x, t)
        f = self.net_f(x, t)
        u = u.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
        return u, f
def plotme(u):
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)

    h = ax.imshow(u.T, interpolation='nearest', cmap='rainbow', 
                  extent=[t.min(), t.max(), x.min(), x.max()], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15) 

    ax.plot(
        X_u_train[:,1], 
        X_u_train[:,0], 
        'kx', label = 'Data (%d points)' % (u_train.shape[0]), 
        markersize = 4,  # marker size doubled
        clip_on = False,
        alpha=1.0
    )

    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)

    ax.set_xlabel('$t$', size=20)
    ax.set_ylabel('$x$', size=20)
    ax.legend(
        loc='upper center', 
        bbox_to_anchor=(0.9, -0.05), 
        ncol=5, 
        frameon=False, 
        prop={'size': 15}
    )
    ax.set_title('$u(t,x)$', fontsize = 20) # font size doubled
    ax.tick_params(labelsize=15)

    plt.show()


# ## All in one general parameter estimator

# In[4]:


# def meshgrid_generator():
data = scipy.io.loadmat('../Burgers Equation/data/burgers_shock.mat')
#t and x are used for a meshgrid:
t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]
X, T = np.meshgrid(x,t)
#     return X, T

# def analytic_solution(X,T):
#     u = X
#     #handle boundary conditions
#     return u

# def analytic_solution(X,T):
#     u = np.sin(T)+np.cos(T)
#     #handle boundary conditions
#     return u

def analytic_solution(X,T):
    u = np.sin(8*X)*np.cos(10*T)
    #handle boundary conditions
    return u


# In[5]:
