# %%

import numpy as np

import scipy.sparse as sp 

from distance import distance  

romes = list(distance.keys())

# %% 

def rho_equiv(mH,mL,b,dHH,dHL):
    rhoH = np.round(((1+(mH*((b-1)/2)**(-1/b))**(-b))/(1+(mL*((b)/2)**(-1/b))**(-b)))**(1/(2*dHH*b)),2)
    rhoL = (((1+(mH*((b-1)/2)**(-1/b))**(-b))/(1+(mL*((b)/2)**(-1/b))**(-b)))**(1/(dHL*b)))/rhoH
    rhoL = np.round(rhoL,2)
    return rhoH,rhoL

def rho_equiv_old(mH,mL,b,d):
    rhoH = np.round(((1+mH**(-b))/(1+mL**(-b)))**(1/(2*d)),2)
    rhoL = np.round(rhoH/2,2)
    return rhoH,rhoL

def get_D(DE,BRANCHES):
    D = np.empty(shape=(len(DE),len(BRANCHES)),dtype=np.int8)
    for n,de in enumerate(DE):
        for b,branch in enumerate(BRANCHES):
            D[n,b] = distance[de.rome][branch.rome]
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
    return np.transpose([[de.T for de in DE]]).astype(np.int8)

def get_H_M(BRANCHES):
    H = np.array([branch.hirings for branch in BRANCHES])
    M0 = np.array([branch.m[0] for branch in BRANCHES])
    M1 = np.array([branch.m[1] for branch in BRANCHES])
    return H, M0, M1

def get_X(distance,DE,BRANCHES):
    X = {}
    #THETA = get_THETA(DE,BRANCHES)
    for n,de in enumerate(DE):
        X[n] = np.empty(shape=(3,len(BRANCHES)))
        for b,branch in enumerate(BRANCHES):
            X[n][0][b] = (1-branch.rho*de.rho)*distance[de.rome][branch.rome]
            X[n][1][b] = branch.m[0]
            X[n][2][b] = branch.hirings
            #X[n][4][b] = LOCAL_THETA[b]
            ##X[n][4][b] = THETA[branch.rome]['U']
            #X[n][5][b] = THETA[branch.rome]['V']
    return X

def interpret(beta):
    result = {}
    #result['d'] = round(beta[0],2)
    result['rhos*d'] = round(beta[0],2)
    result['m'] = round(beta[1],2) 
    #result['h'] = round(beta[2],2)
    return result
    
def get_ALPHA(beta,MAT,DE,BRANCHES,method,beta_default=np.array([-1,0,0])):
    for n in MAT.X.keys():
        np.dot(beta,MAT.X[n],out=MAT.ALPHA[n])
        np.exp(MAT.ALPHA[n],out=MAT.ALPHA[n])
        tot = MAT.ALPHA[n].sum()
        np.divide(MAT.ALPHA[n],tot,out=MAT.ALPHA[n]) 
        if np.isnan(MAT.ALPHA[n]).any() or np.isinf(MAT.ALPHA[n]).any():
            print(f'switched to default beta for de {n}') 
            np.dot(beta_default,MAT.X[n],out=MAT.ALPHA[n])
            np.exp(MAT.ALPHA[n],out=MAT.ALPHA[n])
            np.divide(MAT.ALPHA[n],MAT.ALPHA[n].sum(),out=MAT.ALPHA[n])

            
def get_P(MAT,DE,BRANCHES):
    np.subtract(1,MAT.ALPHA,out=MAT.P)
    np.power(MAT.P,MAT.T,out=MAT.P) 
    np.subtract(1,MAT.P,out=MAT.P)
    #MAT.P = 1 - (1-MAT.ALPHA)**MAT.T

def f(x,a,b,y):
    np.divide(x,a,out=y)
    np.power(y,b,out=y)
    np.add(1,y,out=y)
    np.divide(1,y,out=y) 
    np.power(y,1/b,out=y)
    
def D2f(x,a,b):  
    return (2*(x/a)**b-b+1)/(a**2)


def f_old(x,a,b,y):
    np.divide(x,a,out=y)
    np.power(y,b,out=y)
    np.add(1,y,out=y)
    np.divide(1,y,out=y)    
    
def f_simple(x,a,b):
    return 1/((1+(x/a)**b)**(1/b))

def D2f_old(x,a,b):  
    return (b/(a**2))*((x/a)**(b-2))*((b-1)-(b+1)*(x/a)**b)

def get_PI(MAT,DE,BRANCHES,method):
    np.multiply(MAT.P,MAT.RHO_DE,out=MAT.P)
    np.sum(MAT.P,axis=0,out=MAT.W)
    np.sum(MAT.P*(1-MAT.P),axis=0,out=MAT.V)
    f(MAT.W,MAT.M0,MAT.M1,MAT.PI)
    np.add(MAT.PI,0.5*MAT.V*D2f(MAT.W,MAT.M0,MAT.M1)*MAT.PI**(2*(1+MAT.M1)),out=MAT.PI)    
    #np.add(MAT.PI,0.5*MAT.V*D2f(MAT.W,MAT.M0,MAT.M1)*MAT.PI**3,out=MAT.PI)
                      
def get_matches(beta,MAT,DE,BRANCHES,method):
    get_ALPHA(beta,MAT,DE,BRANCHES,method)
    get_P(MAT,DE,BRANCHES)
    get_PI(MAT,DE,BRANCHES,method)
    return np.sum(MAT.RHO_BRANCHES*MAT.PI*MAT.P)  

def prepare_matrices(DE,BRANCHES):
    T = get_T(DE,BRANCHES)
    H, M0, M1 = get_H_M(BRANCHES)
    D = get_D(DE,BRANCHES)
    RHO_DE = get_RHO_DE(D,DE,BRANCHES)
    RHO_BRANCHES = get_RHO_BRANCHES(D,DE,BRANCHES)
    return D, T, H, M0, M1, RHO_DE, RHO_BRANCHES

class Matrices: 
    def __init__(self,distance,DE,BRANCHES):
        D, T, H, M0, M1, RHO_DE, RHO_BRANCHES = prepare_matrices(DE,BRANCHES)
        self.D = D
        self.T = T
        self.H = H
        #self.M0 = M0*((M1-1)/(M1+1))**(-1/M1)
        self.M0 = M0*((M1-1)/2)**(-1/M1)
        self.M1 = M1
        self.RHO_DE = RHO_DE
        self.RHO_BRANCHES = RHO_BRANCHES
        self.ALPHA = np.zeros(shape=(len(DE),len(BRANCHES)))
        self.P = np.zeros(shape=(len(DE),len(BRANCHES)))
        self.PI = np.zeros(shape=len(BRANCHES))
        self.W = np.zeros(shape=len(BRANCHES))
        self.V = np.zeros(shape=len(BRANCHES))
        
        local_theta = {}
        tot = {n:0.0 for n in range(len(DE))}
        for branch in BRANCHES:
            if branch.rome not in list(local_theta.keys()):
                local_theta[branch.rome] = {'U':0.0,'V':0.0}
            local_theta[branch.rome]['V'] += branch.hirings
        for rome in local_theta.keys():
            for n,de in enumerate(DE):
                tot[n] += de.rho**distance[de.rome][rome]
        for rome in local_theta.keys():
            for n,de in enumerate(DE):
                local_theta[rome]['U'] += de.T*de.rho**distance[de.rome][rome]/tot[n]
            local_theta[rome]['theta'] = local_theta[rome]['U']/local_theta[rome]['V']
        
        self.LOCAL_THETA = np.zeros(shape=len(BRANCHES))
        for b,branch in enumerate(BRANCHES):
               self.LOCAL_THETA[b] = local_theta[branch.rome]['theta']
                   
        self.X = get_X(distance,DE,BRANCHES)
        
        self.M0 = self.M0*self.H*self.LOCAL_THETA
        self.REC = sp.lil_matrix((len(DE),len(BRANCHES)),dtype=int) 
        self.PASS = sp.csr_matrix((len(DE),len(BRANCHES)))
        
class SpMatrices:  
    def __init__(self,distance,DE,BRANCHES):
        self.ALPHA = np.zeros(shape=(len(DE),len(BRANCHES)))
        self.P = np.zeros(shape=(len(DE),len(BRANCHES)))
        self.X = get_X(distance,DE,BRANCHES)
        
                       
class Method: 
    def __init__(self,var=True,opt='Nelder-Mead',correct_alpha=True,draws='recall'):
        self.var = var 
        self.opt = opt 
        self.correct_alpha = correct_alpha 
        self.draws = draws

