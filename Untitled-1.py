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


# In[2]:


# CUDA support 
# if torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
device = torch.device('cpu')


# In[3]:


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


param_vec = np.zeros((9,1))

# #paramvec for u=x:
# param_vec[4,0] = 1
# param_vec[5,0] = 1
# param_vec[7,0] = 1
# param_vec[8,0] = -1
# param_vec = param_vec.T

#paramvec for u=np.sin(T)+np.cos(T):
param_vec[3,0] = 1
param_vec[0,0] = 1
param_vec[7,0] = 1
param_vec[5,0] = 1
param_vec[8,0] = -1
param_vec[6,0] = 1
param_vec = param_vec.T

#paramvec for 2nd order wave eqn.:
# param_vec[2,0] = 1
# param_vec[4,0] = 10
# param_vec[5,0] = 1
# param_vec[7,0] = 0
# param_vec[8,0] = 0
# param_vec = param_vec.T


# In[6]:


u = analytic_solution(X,T)
# plotme(u)


# In[7]:


nu = 0.01/np.pi
noise = 0.0        

N_u = 100
N_f = 10000
layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]


Exact = analytic_solution(X,T)
X, T = np.meshgrid(x,t)

X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_star = Exact.flatten()[:,None]              

# Doman bounds
lb = X_star.min(0)
ub = X_star.max(0)    

#this is what we deal with next:
#IC:
xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T))
uu1 = Exact[0:1,:].T
#BC1:
xx2 = np.hstack((X[:,0:1], T[:,0:1]))
uu2 = Exact[:,0:1] #np.zeros(np.shape(Exact[:,-1:])) #
#BC2:
xx3 = np.hstack((X[:,-1:], T[:,-1:]))
uu3 = Exact[:,-1:] #np.zeros(np.shape(Exact[:,-1:])) #

X_u_train = np.vstack([xx1, xx2, xx3])
X_f_train = lb + (ub-lb)*lhs(2, N_f)
X_f_train = np.vstack((X_f_train, X_u_train))
u_train = np.vstack([uu1, uu2, uu3])

idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
X_u_train = X_u_train[idx, :]
u_train = u_train[idx,:]


# ## Physics-informed Neural Networks

# In[8]:


param_vec


# In[9]:


#cnet input: paramVec 1X100
#cnet output: X_u_train 100X2,u_train 100X1
class collocationNet(torch.nn.Module):
    def __init__(self, layers=None):
        super(collocationNet, self).__init__()
        if layers is None:
            layers = [9, 20, 20, 20, 100]

        # parameters
        self.depth = len(layers) - 1
        print("depth ",self.depth)
        # set up layer order dict
        # self.activation = torch.nn.Sigmoid
        
        layer_list = list()
        for i in range(self.depth - 1): 
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            # layer_list.append(('activation_%d' % i, self.activation()))
            
        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)
        
        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)
        
    def forward(self, pv):
        
        out = torch.zeros(100,2)
        raw_c_points = self.layers(pv.float())
        print("raw c points shape:",raw_c_points.shape)
        out[:50,0]=raw_c_points[0,:50]
        out[50:,1]=raw_c_points[0,50:]
        out[50:75,0]=1
        out[75:,0]=-1
        
        return out


# In[10]:


# the deep neural network
class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
        
        # parameters
        self.depth = len(layers) - 1
        
        # set up layer order dict
        self.activation = torch.nn.Tanh
        
        layer_list = list()
        for i in range(self.depth - 1): 
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))
            
        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)
        
        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)
        
    def forward(self, x):
        out = self.layers(x)
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
        self.x_u = torch.tensor(X_u[:, 0:1], requires_grad=True).float().to(device)
        self.t_u = torch.tensor(X_u[:, 1:2], requires_grad=True).float().to(device)
        self.x_f = torch.tensor(X_f[:, 0:1], requires_grad=True).float().to(device)
        self.t_f = torch.tensor(X_f[:, 1:2], requires_grad=True).float().to(device)
        self.u = torch.tensor(u).float().to(device)
        
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
        xt = self.cnet(torch.tensor(pv).double().to(device)) #[x,t] = self.cnet(torch.tensor(pv))
        u = self.dnn(xt) #u = self.dnn(torch.cat([x, t], dim=1))
        return u
    
    def val_net_u(self, x, t):  
        u = self.dnn(torch.cat([x, t], dim=1))
        return u

    def net_f(self, x, t):
        """ The pytorch autograd version of calculating residual """
        pv = self.pv
        u = self.val_net_u(x, t)
        
        u_t = torch.autograd.grad(
            u, t, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x, 
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]
        u_tt = torch.autograd.grad(
            u_t, t, 
            grad_outputs=torch.ones_like(u_t),
            retain_graph=True,
            create_graph=True
        )[0]
        u_xt = torch.autograd.grad(
            u_x, t, 
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]
        f = pv[:,0]*u+pv[:,1]*u_t + pv[:,2]*u_x + pv[:,3]*u_tt + pv[:,4]*u_xx
        return f
    
    def loss_func(self):
        self.optimizer.zero_grad()
        
        #now we deal with this:
        u_pred = self.net_u(self.pv)#(self.x_u, self.t_u)
        f_pred = self.net_f(self.x_f, self.t_f)
        loss_u = torch.mean((self.u - u_pred) ** 2)
        loss_f = torch.mean(f_pred ** 2)
        
        loss = loss_u + loss_f
        
        loss.backward()
        self.iter += 1
        if self.iter % 100 == 0:
            print(
                'Iter %d, Loss: %.5e, Loss_u: %.5e, Loss_f: %.5e' % (self.iter, loss.item(), loss_u.item(), loss_f.item())
            )
        return loss
    
    def train(self):
        self.dnn.train()
                
        # Backward and optimize
        self.optimizer.step(self.loss_func)

            
    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)

        self.dnn.eval()
        u = self.val_net_u(x, t)
        f = self.net_f(x, t)
        u = u.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
        return u, f


# ## Configurations

# In[12]:


# orig_rig GG
# nu = 0.01/np.pi
# noise = 0.0        

# N_u = 100
# N_f = 10000
# layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

# data = scipy.io.loadmat('data/burgers_shock.mat')

# #t and x are used for a meshgrid:
# t = data['t'].flatten()[:,None]
# x = data['x'].flatten()[:,None]
# #Exact is probably the numerical solution to this equation
# Exact = np.real(data['usol']).T

# X, T = np.meshgrid(x,t)

# X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
# u_star = Exact.flatten()[:,None]              

# # Doman bounds
# lb = X_star.min(0)
# ub = X_star.max(0)    

# xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T))
# uu1 = Exact[0:1,:].T
# xx2 = np.hstack((X[:,0:1], T[:,0:1]))
# uu2 = Exact[:,0:1]
# xx3 = np.hstack((X[:,-1:], T[:,-1:]))
# uu3 = Exact[:,-1:]

# X_u_train = np.vstack([xx1, xx2, xx3])
# X_f_train = lb + (ub-lb)*lhs(2, N_f)
# X_f_train = np.vstack((X_f_train, X_u_train))
# u_train = np.vstack([uu1, uu2, uu3])

# idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
# X_u_train = X_u_train[idx, :]
# u_train = u_train[idx,:]


# ## Training

# In[13]:


pv = param_vec
pv = torch.tensor(pv)#.to(device)
device


# In[14]:


model = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub, nu,param_vec)


# In[15]:


# get_ipython().run_cell_magic('time', '', '               \nmodel.train()')


# In[16]:


np.shape(X_star)


# In[17]:


u_pred, f_pred = model.predict(X_star)

error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
print('Error u: %e' % (error_u))                     

U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
Error = np.abs(Exact - U_pred)


# ## Visualizations

# In[ ]:



""" The aesthetic setting has changed. """

####### Row 0: u(t,x) ##################    

fig = plt.figure(figsize=(9, 5))
ax = fig.add_subplot(111)

h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow', 
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


# In[ ]:


####### Row 1: u(t,x) slices ################## 

""" The aesthetic setting has changed. """

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111)

gs1 = gridspec.GridSpec(1, 3)
gs1.update(top=1-1.0/3.0-0.1, bottom=1.0-2.0/3.0, left=0.1, right=0.9, wspace=0.5)

ax = plt.subplot(gs1[0, 0])
ax.plot(x,Exact[25,:], 'b-', linewidth = 2, label = 'Exact')       
ax.plot(x,U_pred[25,:], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')    
ax.set_title('$t = 0.25$', fontsize = 15)
ax.axis('square')
ax.set_xlim([-1.1,1.1])
ax.set_ylim([-1.1,1.1])

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

ax = plt.subplot(gs1[0, 1])
ax.plot(x,Exact[50,:], 'b-', linewidth = 2, label = 'Exact')       
ax.plot(x,U_pred[50,:], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.axis('square')
ax.set_xlim([-1.1,1.1])
ax.set_ylim([-1.1,1.1])
ax.set_title('$t = 0.50$', fontsize = 15)
ax.legend(
    loc='upper center', 
    bbox_to_anchor=(0.5, -0.15), 
    ncol=5, 
    frameon=False, 
    prop={'size': 15}
)

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

ax = plt.subplot(gs1[0, 2])
ax.plot(x,Exact[75,:], 'b-', linewidth = 2, label = 'Exact')       
ax.plot(x,U_pred[75,:], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.axis('square')
ax.set_xlim([-1.1,1.1])
ax.set_ylim([-1.1,1.1])    
ax.set_title('$t = 0.75$', fontsize = 15)

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

plt.show()