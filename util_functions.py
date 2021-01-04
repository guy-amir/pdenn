import numpy as np
import torch
import matplotlib.pyplot as plt

def getIC(pv,x):
    # IC = np.sin(6*x)
    IC = pv[0,5].cpu()*np.exp(-x**2)
    # plt.plot(x,IC)
    # plt.show()
    # return np.expand_dims(IC.cpu(), axis=1)
    return IC

def getBC1(pv,t):
    # BC = np.sin(7*t)
    BC = pv[0,-1].cpu()*t
    # return np.expand_dims(BC.cpu(), axis=1)
    return BC

def getBC2(pv,t):
    # BC = np.sin(7*t)
    BC = pv[0,-2].cpu()*t
    # return np.expand_dims(BC.cpu(), axis=1)
    return BC

def getPV():
    param_vec = np.zeros((9,1))

    # #paramvec for u=x:
    # param_vec[4,0] = 1
    # param_vec[5,0] = 1
    # param_vec[7,0] = 1
    # param_vec[8,0] = -1
    # param_vec = param_vec.T

    #paramvec for u=np.sin(T)+np.cos(T):
    # param_vec[3,0] = 1
    # param_vec[0,0] = 1
    # param_vec[7,0] = 1
    # param_vec[5,0] = 1
    # param_vec[8,0] = -1
    # param_vec[6,0] = 1
    # param_vec = param_vec.T

    #paramvec for 2nd order wave eqn.:
    # param_vec[2,0] = 1
    # param_vec[4,0] = 10
    # param_vec[5,0] = 1
    # param_vec[7,0] = 0
    # param_vec[8,0] = 0
    # param_vec = param_vec.T

    #paramvec for 2nd order wave eqn.:
    param_vec[3,0] = -1
    param_vec[4,0] = 1
    param_vec[5,0] = 1
    param_vec[-1,0] = 0
    param_vec[-2,0] = 0
    # param_vec = param_vec.T

    return param_vec.T
