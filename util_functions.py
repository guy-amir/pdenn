import numpy as np

def getIC(x):
    IC = np.sin(6*x)
    return np.expand_dims(IC, axis=1)

def getBC1(t):
    BC = np.sin(7*t)
    return np.expand_dims(BC, axis=1)

def getBC2(t):
    BC = np.sin(7*t)
    return np.expand_dims(BC, axis=1)

def getPV():
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

    return param_vec
