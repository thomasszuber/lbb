# %% Useful modules 

import numpy as np

import pandas as pd 

from scipy.optimize import minimize 

from matches import get_matches, Matrices, Method, rho_equiv 
 
from draw_recommendations import recommendations, run_reg, stat_des, init_output, append_output, transform, draw_sparse

from simulate_outcomes import bootstrap, bootstrap_pd 

from format_data import gen_DE, gen_BB 

#from test_fake_data import run_test

from distance import distance 

romes = list(distance.keys())

# %% Test on fake data 

#run_test()  

# %% REAL DATA

sample = 'all'

BB_data = pd.read_csv(f"../../bases/treated_BB_test_{sample}.csv")

BB_data = BB_data.sort_values(by=['BE_id','siret'])

BB_data = BB_data.query(f'rome in {romes}')

DE_data = pd.read_csv(f"../../bases/treated_DE_test_{sample}.csv")

DE_data = DE_data.query(f'rome in {romes}')

BE = DE_data.BE_id.unique().tolist()


# %% Parameters 

t = [8,4]

mH,mL = 1.5,0.5

d = 6

b = 3

comb = [((rho_equiv(mH,mL,b,d)),(rho_equiv(mH,mL,b,d)), 
         ((mL,b),(mH,b))) for b in [3]]

# %%

results = {}

method = Method(correct_alpha=True,draws='recall',opt='bfgs',var=True)

for i, (rho_DE, rho_BB, m) in enumerate(comb):
    
    results[i] = {}
    
    results[i]['param'] = comb[i]
    
    for be in BE: 
        
        print(f'{i+1} for be = {be}')
        
        DE = gen_DE(DE_data.loc[DE_data.BE_id == be],rho_DE,t)
               
        BRANCHES = gen_BB(BB_data.loc[BB_data.BE_id == be],rho_BB,m)
        
        MAT = Matrices(distance,DE,BRANCHES)

        results[i][be] = minimize(lambda beta: -get_matches(beta,MAT,DE,BRANCHES,method),np.zeros(4),method=method.opt) 
        
        print(results[i][be].x)
        

save_results = {v:[] for v in ['BE_id','d','rho*d','m','h',
                               'd_var','rho*d_var','m_var','h_var',
                               'gamma','equiv_d','mH','mL','rhoH','rhoL']
                }



for i, (rho_DE, rho_BB, m) in enumerate(comb):
    
    for be in BE:
        
        save_results['BE_id'].append(be)
        
        save_results['d'].append(results[i][be].x[0])
        save_results['d_var'].append(results[i][be].x[0]/results[i][be].hess_inv[0,0])
        
        save_results['rho*d'].append(results[i][be].x[1])
        save_results['rho*d_var'].append(results[i][be].x[1]/results[i][be].hess_inv[1,1])
        
        save_results['m'].append(results[i][be].x[2])
        save_results['m_var'].append(results[i][be].x[2]/results[i][be].hess_inv[2,2])
        
        save_results['h'].append(results[i][be].x[3])
        save_results['h_var'].append(results[i][be].x[3]/results[i][be].hess_inv[3,3])
        
        save_results['gamma'].append(m[0][1])

        save_results['mL'].append(m[0][0])
        
        save_results['mH'].append(m[1][0])
        
        save_results['equiv_d'].append(d)
        
        save_results['rhoH'].append(rho_BB[0])
        
        save_results['rhoL'].append(rho_BB[1])
        
pd.DataFrame(save_results).to_csv(f"../../bases/results_{sample}.csv",index = False)

t = [8,4]

saved_results = pd.read_csv(f"../../bases/results_{sample}.csv")

median_beta = [saved_results['d'].median(),
         saved_results['rho*d'].median(),
         saved_results['m'].median(),
         saved_results['h'].median()]

param = saved_results[['gamma','equiv_d','mH','mL']].drop_duplicates().reset_index()

results = {}

for i,p in param.iterrows(): 
    
    results[i] = {}
    
    results[i]['param'] = {'gamma' : p.gamma, 'equiv_d' : p.equiv_d, 'mH': p.mH, 'mL':p.mL}
    
    results[i]['betas'] = {row['BE_id']:np.array([row['d'],row['rho*d'],row['m'],row['h']]) for j,row in saved_results.loc[saved_results.gamma == p.gamma].loc[saved_results.equiv_d == p.equiv_d].loc[saved_results.mH == p.mH].loc[saved_results.mL == p.mL].iterrows()}

  
    
method = Method(correct_alpha=False,draws='recall',opt='Nelder-Mead',var=True)


for i in results.keys():

    out = init_output()
    
    m = [(results[i]['param']['mH'],results[i]['param']['gamma']),(results[i]['param']['mL'],results[i]['param']['gamma'])]

    rho_DE = rho_equiv(results[i]['param']['mH'],results[i]['param']['mL'],results[i]['param']['gamma'],results[i]['param']['equiv_d'])
    
    rho_BB = rho_equiv(results[i]['param']['mH'],results[i]['param']['mL'],results[i]['param']['gamma'],results[i]['param']['equiv_d'])

    #RS, MATS = [], []

    for be in BE:
        
        print(f'{i+1} for be = {be}')
        
        DE = gen_DE(DE_data.loc[DE_data.BE_id == be],rho_DE,t)
               
        BRANCHES = gen_BB(BB_data.loc[BB_data.BE_id == be],rho_BB,m)
        
        MAT = Matrices(distance,DE,BRANCHES)
    
        #MATS.append(MAT)
          
        R = recommendations(results[i]['betas'][be],MAT,DE,BRANCHES,method,beta_default=median_beta)
    
        #print(stat_des(R,DE,BRANCHES,m,rho_DE,rho_BB))

        #RS.append(R)
    
        out = append_output(out,R,MAT,DE,BRANCHES) 
        
    transform(out)
    
    pd.DataFrame(out).to_csv(f"../../bases/out_{sample}_{i}.csv",index = False)
        
    #run_reg(results[i],out)
    
    #results[i]['beta_boot'],results[i]['var_boot'] = bootstrap_pd(out,nb_iter=10,true_param=[0.5,0.5,0.5])
        
    
    # %%

for i in results.keys():
    print(results[i]['param'])
    for be in BE:
        print(be,results[i]['betas'][be])


# %%
for i in results.keys():
    print(i,results[i]['t'])

        
# %%
for i in results.keys():
    print(i,results[i]['beta_boot'])
# %%
for i in results.keys():
    print(i,results[i]['var_boot'])

