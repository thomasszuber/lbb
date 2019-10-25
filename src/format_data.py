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
        BRANCHES.append(Branch(row['rome'],rho_BB[row['rho_BB']],m[row['m_BB']],row['h'],siret=row['siret'],be=row['BE_id'],naf=row['codenaf'],tot_hirings=row['tot_h']))
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

# %%
    
def save_results(results,tH,tL,mL,mH,dHL,dHH,rho_BB,m,BE):
    
    save = {v:[] for v in ['BE_id','rho_d','m','h',
                               'rho_d_var','m_var','h_var',
                               'gamma','tH','tL','dHH','dHL','mH','mL','rhoH','rhoL']}

    for be in BE:
        save['BE_id'].append(be)
        save['rho_d'].append(results[be].x[0])
        save['rho_d_var'].append(results[be].x[0]/results[be].hess_inv[0,0])
        save['m'].append(results[be].x[1])
        save['m_var'].append(results[be].x[1]/results[be].hess_inv[1,1])
        save['h'].append(results[be].x[2])
        save['h_var'].append(results[be].x[2]/results[be].hess_inv[2,2])
        for v in ['gamma','tH','tL','dHH','dHL','mH','mL','rhoH','rhoL']:
            save[v].append(results['param'][v])
    return save 
    

