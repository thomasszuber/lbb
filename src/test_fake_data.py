# %%
import numpy as  np 

import numpy.random as rd 

from matches import rho_equiv, get_matches, Method, Matrices, interpret  

import matches as ma

from classes import JobSeeker, random_branch

from distance import distance 

from draw_recommendations import recommendations, stat_des, draw_sparse 

from scipy.optimize import minimize 

from simulate_outcomes import simulate_results, bootstrap 

romes = list(distance.keys())

# %%

rd.seed(123)

N = 499

F = 500

hirings_per_branches = 4.5

d, b, mH, mL = 6, 3, 1.5, 0.5

rhoH,rhoL = rho_equiv(mH,mL,b,d)

m = [(mH,b),(mL,b)]

rho_DE =[rhoH,rhoL]

rho_BB =[rhoH,rhoL]

t = [8,4]

DE = [JobSeeker(rd.choice(romes),rd.choice(rho_DE),rd.choice(t),n) for n in range(N)]

BRANCHES = [random_branch(romes,rho_BB,m,hirings_per_branches) for f in range(F)]

MAT = Matrices(distance,DE,BRANCHES)

# %%

method = Method(correct_alpha=True,draws='recall',opt='BFGS')

res = minimize(lambda beta: -get_matches(beta,MAT,DE,BRANCHES,method),np.zeros(4),method=method.opt)

res.x

# %%

R = recommendations(res.x,MAT,DE,BRANCHES,method)
    
print(stat_des(R,DE,BRANCHES,m,rho_DE,rho_BB))

# %%

# %%

R = draw_sparse(res.x,DE,BRANCHES)
    
print(stat_des(R,DE,BRANCHES,m,rho_DE,rho_BB))


# %%

def run_test():
    
    ma.get_ALPHA(res.x,MAT,DE,BRANCHES,method)

    MAT.ALPHA.shape == (N,F)

    MAT.ALPHA.sum(axis=1).min() > 0.99999

    ma.get_P(MAT,DE,BRANCHES)

    t1 = (MAT.P == (1-(1-MAT.ALPHA)**MAT.T)).all()

    np.multiply(MAT.P,MAT.RHO_DE,out=MAT.P)

    np.sum(MAT.P,axis=0,out=MAT.W)

    t2 = MAT.W.shape == (len(BRANCHES),)

    np.sum(MAT.P*(1-MAT.P),axis=0,out=MAT.V)

    MAT.V.shape == (len(BRANCHES),)
    
    ma.f(MAT.W,MAT.M0,MAT.M1,MAT.PI)

    t3 = (MAT.PI == 1/(1+(MAT.W/MAT.M0)**MAT.M1)).all()

    t4 = MAT.PI.shape == (len(BRANCHES),)

    np.add(MAT.PI,0.5*MAT.V*ma.D2f(MAT.W,MAT.M0,MAT.M1)*MAT.PI**3,out=MAT.PI)

    t5 = np.sum(MAT.RHO_BRANCHES*MAT.PI*MAT.P) == 722.5188226771178
    
    t5 = list(interpret(res.x).values()) == [-3.4, 2.6, 1.23, 0.08]
    
    return all([t1,t2,t3,t4,t5])

# %% 

#beta, var = bootstrap([R],[MAT],nb_iter=10,true_param=[0.7,0.2,10])
    

