# %% Useful modules 

import numpy as np

import numpy.random as rd

import pandas as pd 

from matches import get_ALPHA, get_P

from distance import distance

import statsmodels.formula.api as sm

# %%

def draws_without_recall(objects,prob,nb_draws):
    draws = []
    balls = objects
    pi = prob
    for d in range(nb_draws):
        draws += [rd.choice(balls,p=pi)]
        pi = [pi[p] for p,b in enumerate(balls) if b != draws[d]]
        balls = [b for b in balls if b != draws[d]]
        pi = [p/sum(pi) for p in pi]
    return draws

def recommendations(beta,MAT,DE,BRANCHES,method,seed=123,beta_default=np.array([-1,0,0,0])):
    rd.seed(seed)
    R = {}
    get_ALPHA(beta,MAT,DE,BRANCHES,method,beta_default=beta_default)
    get_P(MAT,DE,BRANCHES)
    branches = list(range(len(BRANCHES)))
    if method.draws == 'norecall':
        for n,de in enumerate(DE):
            R[n] = draws_without_recall(branches,MAT.ALPHA[n],de.T)
    if method.draws == 'rank':
        for n,de in enumerate(DE):
            R[n] = MAT.ALPHA[n].argsort()[-de.T:][::-1]
    if method.draws == 'recall':
        for n,de in enumerate(DE):
            R[n] = rd.choice(branches,p=MAT.ALPHA[n],size=de.T)
    for n,r in R.items():
        for b in r:
            MAT.REC[n,b] = 1
    return R

def draw_sparse(beta,DE,BRANCHES,max_branches=100,prob=False,beta_default=np.array([-1,0,0,0]),seed=123):
    rd.seed(seed)
    R = {}
    if prob == True:
        P = {}
    for n,de in enumerate(DE):
        X = np.empty(shape=(4,len(BRANCHES)))
        for b,branch in enumerate(BRANCHES):
            X[0][b] = distance[de.rome][branch.rome]
            X[1][b] = branch.rho*de.rho*distance[de.rome][branch.rome]
            X[2][b] = branch.m[0]
            X[3][b] = branch.hirings
        alpha = np.dot(beta,X)
        np.exp(alpha,out=alpha)
        if len(BRANCHES) > max_branches:
            branches = alpha.argsort()[-max_branches::]
            alpha = alpha[branches]
        else:
            branches = list(range(len(BRANCHES)))
        tot = alpha.sum()
        np.divide(alpha,tot,out=alpha) 
        if np.isnan(alpha).any() or np.isinf(alpha).any():
            print(f'switched to default beta for de {n}')
            alpha = np.dot(beta_default,X) # alpha needs reshaping because it may have been shrunk to max_branches
            np.exp(alpha,out=alpha)
            if len(BRANCHES) > max_branches:
                branches = alpha.argsort()[-max_branches::]
                alpha = alpha[branches]
            else:
                branches = list(range(len(BRANCHES)))
            tot = alpha.sum()
            np.divide(alpha,tot,out=alpha)
        R[n] = rd.choice(branches,p=alpha,size=de.T)
        if prob == True:
            P[n] = {bb:alpha[np.where(branches == bb)][0] for bb in set(R[n])}
    if prob == True:
        return R, P
    else:
        return R 
        

def stat_des(R,DE,BRANCHES,m_BB,rho_DE,rho_BB):
    
    S = {}

    T_BB = {b:0 for b in range(len(BRANCHES))}
    for n,r in R.items():
        for b in r:
            T_BB[b] += 1
    
    S = {'d_DE':{rho:[0.0,sum([int(de.rho == rho)*de.T for de in DE])] for rho in rho_DE},
         'd_BB':{rho:[0.0,sum([int(branch.rho == rho)*T_BB[b] for b,branch in enumerate(BRANCHES)])] for rho in rho_BB},
         'nb_BB':{m:[0.0,sum([int(branch.m == m) for branch in BRANCHES])] for m in m_BB}}
    
    for n,r in R.items():
        for b in r:
            d = distance[DE[n].rome][BRANCHES[b].rome]
            S['d_DE'][DE[n].rho][0] += d/S['d_DE'][DE[n].rho][1]
            S['d_BB'][BRANCHES[b].rho][0] += d/S['d_BB'][BRANCHES[b].rho][1]
    
    for b,T in T_BB.items():
        S['nb_BB'][BRANCHES[b].m][0] += T/(BRANCHES[b].hirings*S['nb_BB'][BRANCHES[b].m][1])
    
    S = {'d_DE':{rho:list(np.round(S['d_DE'][rho],2)) for rho in rho_DE},
         'd_BB':{rho:list(np.round(S['d_BB'][rho],2)) for rho in rho_BB},
         'nb_BB':{m:list(np.round(S['nb_BB'][m],2)) for m in m_BB}}
    
    return S

def init_output():
    out = {'bni':[],'rome_DE':[],'siret':[],'naf':[],'rome_BB':[],'h':[], 
           'd':[],'rho_DE':[],'T_DE':[],'rec':[],'pi':[],'rho_BB':[],'m_BB':[],'BE_id':[],
           'd_DE':[],'d_BB':[],'tot_h':[],'R':[]}
    return pd.DataFrame(out) 
    
def append_output(previous,R,MAT,DE,BRANCHES):
    out = {'bni':[],'rome_DE':[],'siret':[],'naf':[],'rome_BB':[],'h':[], 
           'd':[],'rho_DE':[],'T_DE':[],'rec':[],'pi':[],'rho_BB':[],'m_BB':[],'BE_id':[]}
    for n,r in R.items():
        out['bni'] += [DE[n].bni]*len(r)
        out['rome_DE'] += [DE[n].rome]*len(r)
        out['rho_DE'] += [DE[n].rho]*len(r)
        out['T_DE'] += [DE[n].T]*len(r)
        out['BE_id'] += [DE[n].be]*len(r)
        out['rec'] += [1]*len(r)
        for b in r:
            out['siret'].append(BRANCHES[b].siret)
            out['rome_BB'].append(BRANCHES[b].rome)
            out['d'].append(distance[DE[n].rome][BRANCHES[b].rome])
            out['rho_BB'].append(BRANCHES[b].rho)
            out['m_BB'].append(BRANCHES[b].m[0])
            out['h'].append(BRANCHES[b].hirings) 
            out['naf'].append(BRANCHES[b].naf)
            out['pi'].append(MAT.ALPHA[n][b])
    
    out = pd.DataFrame(out)
    
    out['d_DE'] = out['d'].groupby(out['bni']).transform('mean')
    
    out['d_BB'] = out['d'].groupby(out['siret']).transform('mean')
    
    out['tot_h'] = out['h'].groupby(out['siret']).transform('sum')

    out['R'] = out['rec'].groupby(out['siret']).transform('sum')/out['tot_h']
    
    return previous.append(out,ignore_index=True,sort=False) 



def fill_output(RS,MATS,DES,BRANCHESS):
    out = {'bni':[],'rome_DE':[],'siret':[],'naf':[],'rome_BB':[],'h':[], 
           'd':[],'rho_DE':[],'T_DE':[],'rec':[],'pi':[],'rho_BB':[],'m_BB':[],'BE_id':[]}
    for R,MAT,DE,BRANCHES in zip(RS,MATS,DES,BRANCHESS):
        for n,r in R.items():
            out['bni'] += [DE[n].bni]*len(r)
            out['rome_DE'] += [DE[n].rome]*len(r)
            out['rho_DE'] += [DE[n].rho]*len(r)
            out['T_DE'] += [DE[n].T]*len(r)
            out['BE_id'] += [DE[n].be]*len(r)
            out['rec'] += [1]*len(r)
            for b in r:
                out['siret'].append(BRANCHES[b].siret)
                out['rome_BB'].append(BRANCHES[b].rome)
                out['d'].append(distance[DE[n].rome][BRANCHES[b].rome])
                out['rho_BB'].append(BRANCHES[b].rho)
                out['m_BB'].append(BRANCHES[b].m[0])
                out['h'].append(BRANCHES[b].hirings) 
                out['naf'].append(BRANCHES[b].naf)
                out['pi'].append(MAT.P[n][b])
    
    out = pd.DataFrame(out)
    
    return out

def transform(out): 
    
    out['d_DE'] = out['d'].groupby(out['bni']).transform('mean')
    
    out['d_BB'] = out['d'].groupby(out['siret']).transform('mean')
    
    out['tot_h'] = out['h'].groupby(out['siret']).transform('sum')

    out['R'] = out['rec'].groupby([out['siret']]).transform('sum')
    
    out['rh'] = out['R']/out['tot_h']
    


def run_reg(results,out):
    
    results['m_BB'] = sm.ols(formula="R ~ m_BB*tot_h ", data=out[['siret','R','tot_h','m_BB']].drop_duplicates()).fit()
    
    results['rho_BB'] = sm.ols(formula="d ~ rho_BB", data=out).fit()
    
    results['rho_DE'] = sm.ols(formula="d ~ rho_DE", data=out).fit()
    
    #results['T_DE'] = sm.ols(formula="DE_hired ~ T_DE + C(BE_id)", data=out[['bni','DE_hired','d_DE','T_DE','rome_DE','BE_id']].drop_duplicates()).fit()
    
    #results['R'] = sm.ols(formula="BB_hires ~ R + h + C(BE_id)", data=out[['siret','BB_hires','d_BB','R','tot_h','rome_BB','BE_id']].drop_duplicates()).fit()
    
    
    results['t'] = {v:results[v].tvalues[v] for v in ['rho_DE','rho_BB','m_BB']}
    
    results['R2'] = {v:results[v].rsquared for v in ['rho_DE','rho_BB','m_BB']}
    
    #out['R_d'] = out['rec'].groupby([out['siret'],out['d']]).transform('sum')/out['tot_h']
    
    #out['std_d_DE'] = out['d'].groupby([out['BE_id'],out['rome_DE']]).transform('std')
    
    #out['mean_d_DE'] = out['d'].groupby([out['BE_id'],out['rome_DE']]).transform('mean')
    
    #out['std_d_BB'] = out['d'].groupby([out['BE_id'],out['naf']]).transform('std')
    
    #out['mean_d_BB'] = out['d'].groupby([out['BE_id'],out['naf']]).transform('mean')
    
    #out['std_d_BB_within'] = out['d'].groupby([out['BE_id'],out['naf'],out['siret']]).transform('std')
    
    #out['std_R'] = out[['siret','R','BE_id','naf']].drop_duplicates().R.groupby([out['BE_id'],out['naf']]).transform('std')
    
    #out['mean_R'] = out[['siret','R','BE_id','naf']].drop_duplicates().R.groupby([out['BE_id'],out['naf']]).transform('mean')
    
    #out['std_R_d'] = out[['siret','R_d','d','BE_id','naf']].drop_duplicates().R_d.groupby([out['BE_id'],out['naf'],out['d']]).transform('std')

    #out['mean_R_d'] = out[['siret','R_d','d','BE_id','naf']].drop_duplicates().R_d.groupby([out['BE_id'],out['naf'],out['d']]).transform('mean')
    
    #sum_treat = {'d_DE' : [out[['BE_id','rome_DE','mean_d_DE']].drop_duplicates().mean_d_DE.median(),
     #                    out[['BE_id','rome_DE','std_d_DE']].drop_duplicates().std_d_DE.median()],
      #         'd_BB' : [out[['BE_id','naf','mean_d_BB']].drop_duplicates().mean_d_BB.median(),
       #                  out[['BE_id','naf','std_d_BB']].drop_duplicates().std_d_BB.median()],
        #       'R_bb' : [out[['BE_id','naf','mean_R']].drop_duplicates().mean_R.median(),
         #                out[['BE_id','naf','std_R']].drop_duplicates().std_R.median()],
          #     'R_bb_d' : [out[['BE_id','naf','d','mean_R_d']].drop_duplicates().groupby(['d']).mean_R_d.median(),
           #                out[['BE_id','naf','d','std_R_d']].drop_duplicates().groupby(['d']).std_R_d.median()]
           #   }
    
    #results['sum_treat'] = sum_treat