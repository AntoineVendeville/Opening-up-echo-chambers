import gurobi as gp
import numpy as np



def gini(x, normalise=True, S=5):
    if normalise:
        vect = np.full(S,1/S)
        gini_max = gini(vect,False,S)
    else:
        gini_max = 1
    return x.dot(1-x)/gini_max

def maxGini_ineq(N,S,Lambda,Mu,Leaders,B,p_init=None,penalty=False,gamma1=1,gamma2=10000,weighted=False,eps=.0001,method=-1,verbose=False,numfocus=0):
    m = gp.Model()
    if not verbose:
        m.setParam('OutputFlag', 0)
    m.setParam('Method',method)
    m.setParam('NumericFocus',numfocus)

    print('B:',B)
    print('numfocus', numfocus)
    print('penalty:', penalty) 
  
    # add p variables and objective
    p = m.addMVar((N,S), lb=0)
    nu = m.addMVar((N,S), lb=0)
    gini = 1 - sum([p[n,s]**2 for n in range(N) for s in range(S)]) /N
    objective = 1-gini
    if penalty:
        if weighted:
            penalty1 = sum([((p[n,s]-p_init[n,s])/(p_init[n,s]+eps))**2 for n in range(N) for s in range(S)]) /N
        else:
            penalty1 = sum([(p[n,s]-p_init[n,s])**2 for n in range(N) for s in range(S)]) /N
        objective += gamma1*penalty1
    penalty2 = 0
    for n in range(N):
        lead = Leaders[n]
        lead_posting_rate = Lambda[lead].sum() + Mu[lead].sum()
        penalty2 += ( sum([nu[n,s] for s in range(S)]) - B/(1-B)*lead_posting_rate )**2
    objective += gamma2*penalty2
    m.setObjective(objective)

    # constraints
    for n in range(N):
        lead = Leaders[n]
        lead_posting_rate = Lambda[lead].sum() + Mu[lead].sum()
        m.addConstr(nu[n,:].sum()<=B/(1-B)*lead_posting_rate) # constraint on \nu^{(n)}
        for s in range(S): # balance equations
            m.addConstr(p[n,s]/(1-B)*lead_posting_rate
                        == nu[n,s] + sum([Lambda[k,s]+Mu[k]*p[k,s] for k in lead]))
    
    # solve
    m.update()
    #print(m.display())
    m.optimize()
    m.update()
    return (nu.X, p.X, m.Runtime)
