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

# from model import *
from util_functions import *

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
        self.activation = torch.nn.Tanh()
        
    def forward(self, pv):
        pv = pv.float()
        # out = self.activation(self.fc1(pv))
        # out = self.activation(self.fc2(out))
        # out = self.activation(self.fc3(out))
        # out = self.activation(self.fc4(out))
        # out = self.activation(self.fc5(out))
        # out = self.activation(self.fc6(out))
        # out = self.activation(self.fc7(out))
        # out = self.fc8(out)
        out = self.fc1(pv)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)
        out = self.fc6(out)
        out = self.fc7(out)
        out = self.fc8(out)
        
        return out

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
        self.activation = torch.nn.Tanh()      

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

class PhysicsInformedNN():
    def __init__(self, X_u, u, X_f, layers, lb, ub,pv):
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

        self.dnn = DNN(layers).to(device)
        self.cnet = collocationNet().to(device)

        self.optimizer1 = torch.optim.SGD(self.dnn.parameters(), lr=0.3)
        self.optimizer2 = torch.optim.SGD(self.cnet.parameters(), lr=0.3)

    def prep_cp(self):
        xt = self.cnet(self.pv)
        x = xt.t()
        
        t = xt.clone().t()
        x = 2*(x-x.min())/(x.max()-x.min())-1
        t = (t-t.min())/(t.max()-t.min())
        # x = torch.ones(100,1)
        x[50:75,0] = 1
        x[75:,0] = -1
        # x[:50,0] = xt[0,:50]
        # t = torch.zeros(100,1)
        
        t[:50,0] = 0
        # t[50:,0] = xt[0,50:]
        return x, t

    def net_u(self, x, t):  
        u = self.dnn(torch.cat([x, t], dim=1))
        return u

    def net_f(self, x, t):
        """ The pytorch autograd version of calculating residual """
        u = self.net_u(x, t)
        
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
    
    def loss_func(self, u_gt, u_pred, f_pred):

        loss_u = torch.mean((torch.tensor(u_gt).to(device) - u_pred) ** 2)
        loss_f = 100*torch.mean(f_pred ** 2)
        
        loss = loss_u + loss_f

        return loss_u

    def train(self):
        self.dnn.train()
    
    def eval(self):
        self.dnn.eval()

    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
        # self.cnet.eval()
        self.dnn.eval()
        u = self.net_u(x, t)
        f = self.net_f(x, t)
        u = u.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
        return u, f

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# Exact = analytic_solution(X,T)
t = torch.linspace(0,1,100)
x = torch.linspace(-1,1,256)
X, T = torch.meshgrid(x,t)

#X_star retresents a list of all (X,T) pairs on the grid:
X_star = torch.cat((X.flatten()[:,None], T.flatten()[:,None]), dim=1) 

# Doman bounds
lb = X_star.min(0)[0]
ub = X_star.max(0)[0]  

#Boundries:

pv = torch.tensor(getPV()).to(device)

#IC:
xx1 = torch.cat((X[:,0:1], T[:,0:1]),dim=1)
IC = getIC(pv,x)
#BC1:
xx2 = torch.cat((X[0:1,:], T[0:1,:]),dim=0).T
BC1 = getBC1(pv,t) #torch.zeros(torch.shape(Exact[:,-1:])) #
#BC2:
xx3 = torch.cat((X[-1:,:], T[-1:,:]),dim=0).T
BC2 = getBC2(pv,t) #torch.zeros(torch.shape(Exact[:,-1:])) #

pv = torch.tensor(getPV()).to(device)

##preliminary data:
nu = 0.01/np.pi
noise = 0.0        

N_u = 100
N_f = 10000
layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

epochs = 1000
eval_every = 100

X_u_train = torch.cat([xx1, xx2, xx3],dim=0) #all IC and BC x & t
X_f_train = lb + (ub-lb)*lhs(2, N_f)
X_f_train = torch.cat((X_f_train, X_u_train),dim=0)
boundary_ground_truth = torch.cat([IC, BC1, BC2],dim=0) #values on IC & BC

#random shuffleing of BC and IC coordiantes:
idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)

X_u_train = X_u_train[idx, :]
u_train = boundary_ground_truth[idx]
#####

torch.random.manual_seed(1234)
np.random.seed(1234)

net = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub,pv)

for epoch in range(epochs):
    # set models to train mode
    net.train()
    net.optimizer2.zero_grad()
    net.optimizer1.zero_grad()
    xi,ti = net.prep_cp()
    u_pred = net.net_u(xi, ti)

    #IC:
    IC = getIC(pv,xi[:50].detach().cpu().numpy())
    #BC1:
    BC1 = getBC1(pv,ti[50:75].detach().cpu().numpy())
    #BC2:
    BC2 = getBC2(pv,ti[75:].detach().cpu().numpy())
    u_gt = torch.cat([IC, BC1, BC2],dim=0)

    # loss = torch.mean((net.u - u_pred) ** 2)
    # loss.backward()
    

    # f_pred = net.net_u(xi, ti)
    torch.autograd.set_detect_anomaly(True)
    f_pred = net.net_f(net.x_f, net.t_f)
    
    loss = net.loss_func(u_gt,u_pred,f_pred)
    # loss = torch.mean(f_pred ** 2)
    loss.backward()
    net.optimizer2.step()
    net.optimizer1.step()
    
    if epoch % eval_every == eval_every-1:
            # bring models to evaluation mode
            net.eval()
            print(epoch, loss)
print("hi")

u_pred, f_pred = net.predict(X_star)
U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
# Error = np.abs(Exact - U_pred)
xx, tt = net.prep_cp()

fig = plt.figure(figsize=(9, 5))
ax = fig.add_subplot(111)

h = ax.imshow(U_pred, interpolation='nearest', cmap='rainbow', 
              extent=[t.min(), t.max(), x.min(), x.max()], 
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h, cax=cax)
cbar.ax.tick_params(labelsize=15) 

ax.plot(
    tt.detach().cpu().numpy(), 
    xx.detach().cpu().numpy(), 
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
print("hi")