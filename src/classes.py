import numpy.random as rd

class JobSeeker: 
    def __init__(self,rome,rho,T,bni='.',be='.'):
        self.rome = rome 
        self.rho = rho
        self.T = T
        self.bni = bni
        self.be = be
    def theta(self,theta):
        self.theta = theta
        
class Branch: 
    def __init__(self,rome,rho,m,hirings,siret='.',nb='.',be='.',naf='.'):
        self.rho = rho
        self.m = m 
        self.hirings = hirings
        self.rome = rome
        self.siret = siret
        self.nb = nb
        self.be = be
        self.naf = naf
        self.tot_hirings = None 
    def theta(self,theta):
        self.theta = theta     
        
class Firm: 
    def __init__(self,branches):
        self.rho = branches[0].rho
        self.m = branches[0].m 
        self.branches = branches
        self.romes = [b.rome for b in branches]
        self.nb_branches = len(branches)
        self.tot_hirings = sum([b.hirings for b in branches]) 
        self.siret = branches[0].siret
        self.be = branches[0].be
        self.naf = branches[0].naf
        
    def add_branch(self,branch):
        self.branches.append(branch)
        self.romes.append(branch.rome)
        self.tot_hirings += branch.hirings
        self.nb_branches += 1

def random_branch(romes,rho,m,mean_hirings,siret='.',nb='.'): 
    return Branch(rd.choice(romes),rd.choice(rho),m[rd.choice(range(len(m)))],rd.chisquare(mean_hirings),siret=siret,nb=nb)

def random_firm(romes,rho,m,max_branches,mean_hirings,siret='.'):
    rho_f = rd.choice(rho)
    m_f = m[rd.choice(range(len(m)))]
    NB = rd.randint(1,high=max_branches)
    branches = [random_branch(romes,[rho_f],[m_f],mean_hirings,siret=siret,nb=i) for i in range(NB)]
    return Firm(branches) 