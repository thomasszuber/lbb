# %% Useful modules 

import numpy as np

import pandas as pd

import numpy.random as rd

from scipy.optimize import minimize 

from matches import get_matches, SpMatrices, Matrices, Method, rho_equiv 
 
from draw_recommendations import recommendations, init_output, append_output, append_output_sparse, transform, draw_sparse, test_power

from format_data import gen_DE, gen_BB, save_results 

#from test_fake_data import run_test

from distance import distance 

romes = list(distance.keys())

# %% Test on fake data 

#run_test()  

# %% REAL DATA

rd.seed(123) 

sample = 'all'

BB_data = pd.read_csv(f"../../bases/treated_BB_test_{sample}.csv")

BB_data = BB_data.sort_values(by=['BE_id','siret'])

BB_data = BB_data.query(f'rome in {romes}')

#BB_data['tau'] = 1

BB_shrink = BB_data[['siret','tau']].drop_duplicates() 

nb_BB = BB_shrink.shape[0] 

BB_shrink['draw'] = rd.uniform(size=nb_BB)

BB_shrink = BB_shrink.query('draw < tau')

BB_shrink = BB_shrink.siret.unique().tolist()

BB_data = BB_data.query(f'siret in {BB_shrink}')

DE_data = pd.read_csv(f"../../bases/treated_DE_test_{sample}.csv")

DE_data = DE_data.query(f'rome in {romes}')

nb_DE = DE_data.shape[0] 

DE_data['draw'] = rd.uniform(size=nb_DE)

#DE_data['tau'] = 1

DE_data = DE_data.query('draw < tau')

BE = DE_data.BE_id.unique().tolist()


# %% 

tH, tL = 8, 4 

t = [tH,tL]

mH,mL = 1.5,0.5

dHH = 6

dHL = 3

gamma = 3

rhoH, rhoL = rho_equiv(mH,mL,gamma,dHH,dHL)

rho_BB = (rhoH,rhoL)

rho_DE = (rhoH,rhoL)

m = ((mL,gamma),(mH,gamma))


method = Method(correct_alpha=True,draws='recall',opt='bfgs',var=True)
 
results = {}
    
results['param'] = {'rhoH':rhoH, 'rhoL':rhoL, 'mH':mH, 'mL':mL, 'dHH':dHH, 'dHL':dHL, 'tH':tH, 'tL':tL, 'gamma':gamma}
    
for be in BE: 
        
    print(f'be = {be}')
        
    DE = gen_DE(DE_data.loc[DE_data.BE_id == be],rho_DE,t)
               
    BRANCHES = gen_BB(BB_data.loc[BB_data.BE_id == be],rho_BB,m)
        
    MAT = Matrices(distance,DE,BRANCHES)

    results[be] = minimize(lambda beta: -get_matches(beta,MAT,DE,BRANCHES,method),np.zeros(3),method=method.opt) 
        
    print(results[be].x)

pd.DataFrame(save_results(results,tH,tL,mL,mH,dHL,dHH,rho_BB,m,BE)).to_csv(f"../../bases/results_prod_{sample}.csv",index = False)


sample = 'all'

BB_data = pd.read_csv(f"../../bases/treated_BB_test_{sample}.csv")

BB_data = BB_data.sort_values(by=['BE_id','siret'])

BB_data = BB_data.query(f'rome in {romes}')

DE_data = pd.read_csv(f"../../bases/treated_DE_test_{sample}.csv")

DE_data = DE_data.query(f'rome in {romes}')

BE = DE_data.BE_id.unique().tolist()

saved_results = pd.read_csv(f"../../bases/results_prod_{sample}.csv")

median_beta = [saved_results['rho_d'].median(),
         saved_results['m'].median(),
         saved_results['h'].median()]

mean_beta = [saved_results['rho_d'].mean(),
         saved_results['m'].mean(),
         saved_results['h'].mean()]

param = saved_results[['gamma','tH','tL','dHH','dHL','mH','mL']].drop_duplicates()

[gamma, tH, tL, dHH, dHL, mH, mL] = param.iloc[0].tolist()

m = [(mL,gamma),(mH,gamma)]

rhoH, rhoL = rho_equiv(mH,mL,gamma,dHH,dHL)

t = [int(tH),int(tL)]
 
rho_BB = (rhoH,rhoL)

rho_DE = (rhoH,rhoL)


betas = {}

betas = {row['BE_id']:np.array([row['rho_d'],row['m'],row['h']]) for j,row in saved_results.iterrows()}

method = Method(correct_alpha=False,draws='recall',opt='bfgs',var=True)

out = init_output()
    
for be in BE:
        
    DE = gen_DE(DE_data.loc[DE_data.BE_id == be],rho_DE,t)
               
    BRANCHES = gen_BB(BB_data.loc[BB_data.BE_id == be],rho_BB,m)
    
    print(f'be = {be} with {len(BRANCHES)} BB and {len(DE)} DE.')
    
    #MAT = SpMatrices(distance,DE,BRANCHES)
          
    #R = recommendations(betas[be],MAT,DE,BRANCHES,method,beta_default=median_beta)
    
    #out = append_output(out,R,MAT,DE,BRANCHES) 
          
    R,P =  draw_sparse(betas[be],DE,BRANCHES,max_branches=np.inf,prob=True,beta_default=median_beta,seed=123)
    
    out = append_output_sparse(out,R,P,DE,BRANCHES) 
        
transform(out)
    
print(out.groupby(['rho_DE','rho_BB']).d.describe())
    
print(out[['siret','rh','m_BB']].drop_duplicates().groupby(['m_BB'])['rh'].describe())
    
out.to_csv(f"../../bases/out_prod_{sample}.csv",index = False)

# %% 
out['d'].hist(by=[out['rho_DE'],out['rho_BB']],alpha=0.5)

out['rh'].hist(by=out['m_BB'],alpha=0.5)

# %%
out.groupby(['rho_DE','rho_BB']).d.plot.kde(alpha=0.5,bw_method=0.4)

# %%

out.groupby(['m_BB']).rh.plot.kde(alpha=0.5,bw_method=0.4)

# %% 

share_BB_zeros = (BB_data.siret.unique().size - out.siret.unique().size)/BB_data.siret.unique().size
