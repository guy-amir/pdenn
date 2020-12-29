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

from model import *
from util_functions import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Exact = analytic_solution(X,T)
t = np.linspace(0,1,100)
x = np.linspace(-1,1,256)
X, T = np.meshgrid(x,t)

#X_star retresents a list of all (X,T) pairs on the grid:
X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None])) 

# Doman bounds
lb = X_star.min(0)
ub = X_star.max(0)    

#Boundries:
#IC:
xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T))
IC = getIC(x)
#BC1:
xx2 = np.hstack((X[:,0:1], T[:,0:1]))
BC1 = getBC1(t) #np.zeros(np.shape(Exact[:,-1:])) #
#BC2:
xx3 = np.hstack((X[:,-1:], T[:,-1:]))
BC2 = getBC2(t) #np.zeros(np.shape(Exact[:,-1:])) #

pv = torch.tensor(getPV()).to(device)

##preliminary data:
nu = 0.01/np.pi
noise = 0.0        

N_u = 100
N_f = 10000
layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

epochs = 100

X_u_train = np.vstack([xx1, xx2, xx3]) #all IC and BC x & t
X_f_train = lb + (ub-lb)*lhs(2, N_f)
X_f_train = np.vstack((X_f_train, X_u_train))
u_train = np.vstack([IC, BC1, BC2]) #values on IC & BC

#random shuffleing of BC and IC coordiantes:
idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
X_u_train = X_u_train[idx, :]
u_train = u_train[idx,:]
#####

np.random.seed(1234)


# def train(args, model, device, train_loader, optimizer, epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % args.log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))
#             if args.dry_run:
#                 break

model = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub, nu,pv)
print("hi")
# for epoch in range(1, epochs + 1):
#     train(args, model, device, train_loader, optimizer, epoch)
#     test(model, device, test_loader)
#     scheduler.step()
# for inputs, targets in training_data_loader:
#     optimizer.zero_grad()
    
#     outputs = model(inputs)
#     loss = loss_function(outputs, targets)
#     loss.backward()
    
#     optimizer.step()

model.train()
u_pred, f_pred = model.predict(X_star)

# error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
# print('Error u: %e' % (error_u))                     

U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
# Error = np.abs(Exact - U_pred)

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