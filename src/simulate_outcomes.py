# %% Useful modules 

import numpy as np

import numpy.random as rd

from scipy.optimize import minimize 

from scipy.optimize import Bounds 

import scipy.sparse as sp

import statsmodels.formula.api as sm

from matches import f_simple, f

# %% NON PARAMETRIC 

def simulate_results_pd(out,param=[0.5,0.2,3],seed=123):
    
    rd.seed(seed)
    
    l = out.shape[0]
    
    # True Paramaters 
    
    rho_DE, rho_BB, m1 = param 
                
    # Draw applications
    
    out['hired'] = (rd.uniform(size=l) < rho_DE**out['d'])*1
    
    # Screening rates for applications
    
    out['hired'] = f_simple(out['hired'].groupby([out['siret'],out['rome_BB']]).transform('sum')/out['h'],1,m1)*out['hired']
    
    # Screen applications 
    
    out['hired'] = (rd.uniform(size=l) < out['hired'])*1
    
    # Hire screened applications
    
    out['hired'] = (rd.uniform(size=l) < out['hired']*rho_BB**out['d'])*1
    
    # Total hires 
    
    out['DE_hired'] = (out['hired'].groupby(out['bni']).transform('sum') > 0)*1
    
    out['BB_hires'] = out['hired']/out['hired'].groupby(out['bni']).transform('sum')
    
    out['BB_hires'] = out['BB_hires'].groupby(out['siret']).transform('sum')
    

def bootstrap_pd(out,nb_iter=100,true_param=[0.7,0.2,3],seed=123):

    rd.seed(seed)

    SEED = rd.randint(1,high=nb_iter*1000,size=nb_iter)

   # boot = {v:[] for v in ['d','R','pi']} 

    boot = {v:[] for v in ['T_DE','R','h']}

    for i,seed in enumerate(SEED):
    
        print(f'{i+1} out of {nb_iter} bootstraps')
    
        simulate_results_pd(out,param=true_param,seed=seed)
    
        #res = sm.ols(formula="hired ~ d + R + pi + C(BE_id)", data=out).fit()

        #for v in ['d','R','pi']:
            
            #boot[v].append(res.params[v])
    
        res = sm.ols(formula="DE_hired ~ T_DE + C(BE_id)", data=out[['bni','DE_hired','d_DE','T_DE','rome_DE','BE_id']].drop_duplicates()).fit()
    
        for v in ['T_DE']:
            
            boot[v].append(res.params[v])
            
        res = sm.ols(formula="BB_hires ~ R + h + C(BE_id)", data=out[['siret','BB_hires','d_BB','R','h','rome_BB','BE_id']].drop_duplicates()).fit()
    
        for v in ['R','h']:
            
            boot[v].append(res.params[v])
        

    return {v:np.mean(betas) for v,betas in boot.items()}, {v:np.var(betas) for v,betas in boot.items()} 

# %% PARAMETRIC 

def simulate_results(RS,MATS,param=[0.5,0.2,3],seed=123):
    
    rd.seed(seed)
    
    # True Paramaters 
    
    rho_DE, rho_BB, m1 = param 
    
    for R,MAT in zip(RS,MATS):
    
        shape = MAT.D.shape
        
        MAT.RHO_BRANCHES = MAT.REC.multiply(rho_BB**MAT.D)
                
        # Draw applications
        
        MAT.PASS = MAT.REC.multiply(MAT.RHO_DE)
        
        MAT.PASS = sp.csr_matrix(rd.uniform(size=shape) < MAT.PASS)
    
        # Screening rates
        
        MAT.PI = MAT.PASS.sum(axis=0)
        
        MAT.PI = np.divide(MAT.PI,MAT.H)
        
        f(MAT.PI,1,m1,MAT.PI)
        
        MAT.PASS = MAT.PASS.multiply(MAT.PI)
    
        # Screen applications 
    
        MAT.PASS = sp.csr_matrix((rd.uniform(size=shape) < MAT.PASS))
    
        # Hire screened applications
        
        MAT.PASS = MAT.PASS.multiply(MAT.RHO_BRANCHES)
    
        MAT.PASS = sp.csr_matrix(rd.uniform(size=shape) < MAT.PASS)  
        

def likelihood(MATS,param=[0.5,0.5,3]):
    
    l = 0
    
    rho_DE, rho_BB, m1= param 
    
    for MAT in MATS:
        
        MAT.RHO_DE = MAT.REC.multiply(rho_DE**MAT.D)
        
        MAT.PI = MAT.RHO_DE.sum(axis=0)
        
        np.divide(MAT.PI,MAT.H,out=MAT.PI)
        
        f(MAT.PI,1,m1,MAT.PI)
    
        MAT.RHO_DE = MAT.RHO_DE.multiply(MAT.PI) 
        
        MAT.RHO_DE = MAT.RHO_DE.multiply(rho_BB**MAT.D)
    
        MAT.RHO_DE = MAT.PASS - MAT.RHO_DE
        
        MAT.RHO_DE = MAT.RHO_DE.power(2)
        
        l += MAT.RHO_DE.multiply(0.5*MAT.P).sum()/np.sum(MAT.P)
    
    return l 

def bootstrap(RS,MATS,nb_iter = 100,true_param=[0.7,0.2,3],seed=123):

    rd.seed(seed)

    SEED = rd.randint(1,high=nb_iter*1000,size=nb_iter)

    bounds = Bounds([0,0,0],[1,1,np.inf])

    beta = {'rho_DE':[],'rho_BB':[],'m1':[]}

    for i,seed in enumerate(SEED):
    
        print(f'{i+1} out of {nb_iter} bootstraps')
    
        simulate_results(RS,MATS,param=true_param,seed=seed)
    
        res = minimize(lambda x: likelihood(MATS,x),true_param,bounds=bounds,method='trust-constr')
    
        beta['rho_DE'].append(res.x[0])
    
        beta['rho_BB'].append(res.x[1])
        
        beta['m1'].append(res.x[2])
        
    var = {}
    for x,betas in beta.items():
        var[x] = np.var(betas)
    return beta, var 
