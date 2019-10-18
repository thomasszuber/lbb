# %% Useful modules 
from classes import JobSeeker, Branch, Firm 

def gen_DE(data,rho_DE,t):
    DE = []
    for i,row in data.iterrows():
        DE.append(JobSeeker(row['rome'],rho_DE[row['rho_DE']],t[row['T_DE']],bni=row['bni'],be=row['BE_id']))
    return DE

def gen_BB(data,rho_BB,m):
    BRANCHES = []
    for i,row in data.iterrows():
        BRANCHES.append(Branch(row['rome'],rho_BB[row['rho_BB']],m[row['m_BB']],row['h'],siret=row['siret'],be=row['BE_id'],naf=row['codenaf']))
    #FIRMS = {}
    #siret = 0 
    #for branch in BRANCHES:
    #    if siret != branch.siret:
    #        i = 0
    #        branch.nb = i
    #        siret = branch.siret
    #        FIRMS[siret] = Firm([branch])
    #    else:
    #        i += 1 
    #        branch.nb = i
    #        FIRMS[siret].add_branch(branch)
    return BRANCHES

