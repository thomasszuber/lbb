# %% Import modules 

import numpy as np

import scipy.sparse as sp 

from distance import distance  

romes = list(distance.keys())

# %% 

def rho_equiv(mH,mL,b,d):
    rhoH = np.round(((1+mH**(-b))/(1+mL**(-b)))**(1/(2*d)),2)
    rhoL = np.round(rhoH/2,2)
    return rhoH,rhoL

def get_D(distance,DE,BRANCHES):
    D = np.empty(shape=(len(DE),len(BRANCHES)),dtype=np.int8)
    for n,de in enumerate(DE):
        for b,branch in enumerate(BRANCHES):
            D[n][b] = distance[de.rome][branch.rome]
    return D

def get_THETA(DE,BRANCHES):
    thetas = {rome:{'U':0.0,'V':0.0} for rome in romes}
    for de in DE:
        thetas[de.rome]['U'] += 1
    for branch in BRANCHES: 
        thetas[branch.rome]['V'] += branch.hirings
    return thetas

def get_RHO_DE(D,DE,BRANCHES):
    RHO_DE = np.transpose(np.repeat([[de.rho for de in DE]],len(BRANCHES),axis=0))**D
    return RHO_DE

def get_RHO_BRANCHES(D,DE,BRANCHES):
    RHO_BRANCHES = np.repeat([[branch.rho for branch in BRANCHES]],len(DE),axis=0)**D
    return RHO_BRANCHES

def get_T(DE,BRANCHES):
    return np.transpose(np.repeat([[de.T for de in DE]],len(BRANCHES),axis=0))

def get_H_M(BRANCHES):
    H = np.array([branch.hirings for branch in BRANCHES])
    M0 = np.array([branch.m[0] for branch in BRANCHES])
    M1 = np.array([branch.m[1] for branch in BRANCHES])
    return H, M0, M1

def get_X(D,DE,BRANCHES):
    X = {}
    #THETA = get_THETA(DE,BRANCHES)
    for n,de in enumerate(DE):
        X[n] = np.empty(shape=(3,len(BRANCHES)))
        for b,branch in enumerate(BRANCHES):
            X[n][0][b] = D[n][b]
            X[n][1][b] = branch.rho*de.rho*D[n][b]
            X[n][2][b] = branch.m[0] 
            ##X[n][4][b] = THETA[branch.rome]['U']
            #X[n][5][b] = THETA[branch.rome]['V']
    return X

def interpret(beta):
    result = {}
    result['d'] = beta[0]
    result['rhos*d'] = beta[1]
    result['m (F)'] = beta[2]
    return result
    
def get_ALPHA(beta,MAT,DE,BRANCHES,method):
    MAT.ALPHA = np.zeros(shape=(len(DE),len(BRANCHES)))
    for n,de in enumerate(DE):
        MAT.ALPHA[n] = np.exp(beta.dot(MAT.X[n]))
        MAT.ALPHA[n] = MAT.ALPHA[n]/np.sum(MAT.ALPHA[n])
        if method.correct_alpha == True and (np.isnan(MAT.ALPHA[n]).any() or np.isinf(MAT.ALPHA[n]).any()):
            MAT.ALPHA[n] = np.exp(-MAT.D[n])/sum(np.exp(-MAT.D[n]))
            #print(f'corrected alpha for DE nb {n}')

def get_P(MAT,DE,BRANCHES):
    MAT.P = 1 - (1-MAT.ALPHA)**MAT.T

def f(x,a,b):
    return 1/(1+(x/a)**b)

def D2f(x,a,b): 
    return (b/(a**2))*((x/a)**(b-2))*((b-1)-(b+1)*(x/a)**b)*(f(x,a,b)**3)

def get_PI(MAT,DE,BRANCHES,method):
    MAT.W = np.sum(MAT.P*MAT.RHO_DE,axis=0)
    if method.var == True:
        MAT.V = np.sum(MAT.P*MAT.RHO_DE*(1-MAT.P*MAT.RHO_DE),axis=0)
        MAT.PI = np.repeat([f(MAT.W,MAT.H*MAT.LOCAL_THETA*MAT.M0,MAT.M1)+0.5*MAT.V*D2f(MAT.W,MAT.H*MAT.LOCAL_THETA*MAT.M0,MAT.M1)],len(DE),axis=0)
    elif method.var == False:
        MAT.PI = np.repeat([f(MAT.W,MAT.H*MAT.LOCAL_THETA*MAT.M0,MAT.M1)],len(DE),axis=0)   
                      
def get_matches(beta,MAT,DE,BRANCHES,method):
    get_ALPHA(beta,MAT,DE,BRANCHES,method)
    get_P(MAT,DE,BRANCHES)
    get_PI(MAT,DE,BRANCHES,method)
    return np.sum(MAT.RHO_BRANCHES*MAT.PI*MAT.P) 

def prepare_matrices(distance,DE,BRANCHES):
    T = get_T(DE,BRANCHES)
    H, M0, M1 = get_H_M(BRANCHES)
    D = get_D(distance,DE,BRANCHES)
    RHO_DE = get_RHO_DE(D,DE,BRANCHES)
    RHO_BRANCHES = get_RHO_BRANCHES(D,DE,BRANCHES)
    X = get_X(D,DE,BRANCHES)
    return D, T, H, M0, M1, RHO_DE, RHO_BRANCHES, X 

class Matrices: 
    def __init__(self,distance,DE,BRANCHES):
        D, T, H, M0, M1, RHO_DE, RHO_BRANCHES, X = prepare_matrices(distance,DE,BRANCHES)
        self.D = D
        self.T = T
        self.H = H
        self.M0 = M0*((M1-1)/(M1+1))**(-1/M1)
        self.M1 = M1
        self.RHO_DE = RHO_DE
        self.RHO_BRANCHES = RHO_BRANCHES
        self.X = X
        self.theta = sum([de.T for de in DE])/np.sum([b.hirings for b in BRANCHES])
        self.ALPHA = np.zeros(shape=(len(DE),len(BRANCHES)))
        self.P = np.zeros(shape=(len(DE),len(BRANCHES)))
        self.PI = np.zeros(shape=(len(DE),len(BRANCHES)))
        self.W = np.zeros(len(BRANCHES))
        self.local_theta = {}
        
        tot = {n:0.0 for n in range(len(DE))}
        for branch in BRANCHES:
            if branch.rome not in list(self.local_theta.keys()):
                self.local_theta[branch.rome] = {'U':0.0,'V':0.0}
            self.local_theta[branch.rome]['V'] += branch.hirings
        for rome in self.local_theta.keys():
            for n,de in enumerate(DE):
                tot[n] += de.rho**distance[de.rome][rome]
        for rome in self.local_theta.keys():
            for n,de in enumerate(DE):
                self.local_theta[rome]['U'] += de.T*de.rho**distance[de.rome][rome]/tot[n]
            self.local_theta[rome]['theta'] = self.local_theta[rome]['U']/self.local_theta[rome]['V']
        
        self.LOCAL_THETA = np.zeros(len(BRANCHES))
        for b,branch in enumerate(BRANCHES):
               self.LOCAL_THETA[b] = self.local_theta[branch.rome]['theta']
                
        self.REC = np.zeros(shape=(len(DE),len(BRANCHES))) 
        self.PASS = np.zeros(shape=(len(DE),len(BRANCHES))) 
                       
class Method: 
    def __init__(self,var=True,opt='Nelder-Mead',correct_alpha=True,draws='recall'):
        self.var = var 
        self.opt = opt 
        self.correct_alpha = correct_alpha 
        self.draws = draws