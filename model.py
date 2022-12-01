import numpy as np

def gini(x, normalise=True, S=5):
    if normalise:
        vect = np.full(S,1/S)
        gini_max = gini(vect,False,S)
    else:
        gini_max = 1
    return x.dot(1-x)/gini_max


def som_sparse(Lvec,Mvec,Lead):
    Som = dict()
    for user in Lvec:
        Som[user] = 0
        for leader in Lead[user]:
            Som[user] += Lvec[leader] + Mvec[leader]
    return Som

def fill_A_sparse(catf,Lvec,Pvec,Mvec,Lead,Som):
    A = dict()
    # We consider that Lead[j] contains the set of leaders of node j.
    #
    for user in Lvec:
        A[user] = dict()
        for leader in Lead[user]:
            A[user][leader] = Mvec[leader] * Pvec[leader][catf] / Som[user]
    return A

def fill_A_trans_sparse(catf,Lvec,Pvec,Mvec,Lead,Som):
    A_trans = dict()
    # This is the A transpose that we will use also later.
    # A_trans is a dictionary. The keys are the columns of matrix form A. 
    # Each key shows the non-zero elements of A for this column.
    # We consider that Lead[j] contains the set of leaders of node j.
    #
    for user in Lvec:
        A_trans[user] = dict()
    for user in Lvec:
        for leader in Lead[user]:
            A_trans[leader][user] = Mvec[leader] * Pvec[leader][catf] / Som[user]
    return A_trans

def fill_C_sparse(catf,Lvec,Pvec,Mvec):
    C = dict()
    for user in Lvec:
        C[user] = 0
        if Lvec[user]+Mvec[user]>0:
            C[user] = Mvec[user]/(Lvec[user]+Mvec[user])*Pvec[user][catf]
    return C

def fill_gf_sparse_v2(catf,Lvec,Cvec,Som,Lead):
    g = dict()
    for user in Lvec:
        g[user] = 0
        for leader in Lead[user]:
            g[user] += Cvec[leader][catf]/Som[user]
    return g

def fill_hf_sparse_v2(catf,Lvec,Cvec,Mvec):
    return {user: Cvec[user][catf]/(Lvec[user]+Mvec[user]) for user in Lvec}

def pi_method_sparse_v2(N,catf,Lvec,Pvec,Cvec,Mvec,Lead,Som,it,eps):
    # v2: This method resolves the fixed-point exploiting vector sparsity.
    #
    #
    A = fill_A_sparse(catf,Lvec,Pvec,Mvec,Lead,Som)
    A_trans = fill_A_trans_sparse(catf,Lvec,Pvec,Mvec,Lead,Som)
    #
    gf = fill_gf_sparse_v2(catf,Lvec,Cvec,Som,Lead) 
    #
    # Initialisation (the result should be independent of initialisation vector)
    #
    p_new = gf
    p_old = {}
    #
    normdiff = eps+1
    #
    t = 0
    while (t<it) & (normdiff>eps):
        normdiff = 0
        p_old = p_new.copy()
        p_new = {}
        # We search the lines of A which contain non-zero entries coinciding with the non-zero
        # entries of p_old.
        mlines = set()
        for key in p_old:
            for tutu in A_trans[key]:
                mlines.add(tutu)
            #mlines = mlines.union(set(A_trans[key].keys()))
        #print("p_old",p_old)
        for tutu in gf:
            mlines.add(tutu)
        #mlines = mlines.union(set(bi.keys()))
        #print("mlines",mlines)
        for user in mlines:
            p_new[user] = 0
            for leader in Lead[user]:
                if leader in p_old:
                    p_new[user] += A[user][leader]*p_old[leader]
            if user in gf.keys():
                p_new[user]+=gf[user]
            # Norm 1 criterion:
            #normdiff += abs(p_old[user]-p_new[user])
            # Norm INF criterion:
            if user in p_old.keys():
                if abs(p_old[user]-p_new[user])>normdiff:
                    normdiff = abs(p_old[user]-p_new[user])
            else:
                if abs(p_new[user])>normdiff:
                    normdiff = abs(p_new[user])
        t += 1
        #Tracer()()
        #print("p_new",p_new)
    #
    # print("t=",t,"\n")
    # print("diff_last=",normdiff,"\n")
    return p_new


def modelSolve(N,S,Lambda,Mu,Lead,it=1000,eps=.001,vect=True):
    """ vect if we want to return vectors instead of dicts 
    this time we compute only for S-1 categories and deduce the last one
    we comment Phi and Psi because we actually don't use them"""

    # init
    pNews, qWall = dict(), dict() 
    #Psi, Phi = dict(), dict()
    Cvec,Mvec = Lambda,Mu
    Lvec = {u:Lambda.sum(axis=1)[u] for u in range(N)}
    Som = som_sparse(Lvec,Mvec,Lead)
    Pvec = np.ones((N,S)) # this is for the reposting preferences, we don't use them so it's set to 1
    
    # all but one category
    for catf in range(S-1):
        C = fill_C_sparse(catf,Lvec,Pvec,Mvec)
        pNews[catf] = pi_method_sparse_v2(N,catf,Lvec,Pvec,Cvec,Mvec,Lead,Som,it,eps)
        hf = fill_hf_sparse_v2(catf,Lvec,Cvec,Mvec)
        qWall[catf]={}
        #Phi[catf], Psi[catf] = 0, 0
        for userj in pNews[catf]:
            qWall[catf][userj] = C[userj]*pNews[catf][userj]+hf[userj]
            #Psi[catf] += qWall[catf][userj]
            #Phi[catf] += pNews[catf][userj]
        #Psi[catf] = Psi[catf]/N
        #Phi[catf] = Phi[catf]/N

    # last category (if vect we do it below)
    if not vect:
        pNews[S-1] = {n: 1-sum([pNews[s][n] for s in range(S-1)]) for n in range(N)}
        qWall[S-1] = {n: 1-sum([qWall[s][n] for s in range(S-1)]) for n in range(N)}
        #Phi = {s: np.mean([pNews[s][n] for n in range(N)]) for s in range(S)}
        #Psi = {s: np.mean([qWall[s][n] for n in range(N)]) for s in range(S)}
        to_return = (pNews, qWall)#, Phi, Psi)
    
    # vectorise if wanted
    else:
        P, Q = np.zeros((N,S)), np.zeros((N,S))
        for s in range(S-1):
            for n in range(N):
                P[n,s] = pNews[s][n]
                Q[n,s] = qWall[s][n]
        P[:,-1] = 1-P[:,:-1].sum(axis=1)
        Q[:,-1] = 1-Q[:,:-1].sum(axis=1)
        #Phi = P.mean(axis=0)
        #Psi = Q.mean(axis=0)
        to_return = (P, Q)#, Phi, Psi)
    
    # end
    return to_return



###################### TEST WITH DIFFERENT INIT ###########################

def pi_method_sparse_v21(N,catf,Lvec,Pvec,Cvec,Mvec,Lead,Som,it,eps):
    # v2: This method resolves the fixed-point exploiting vector sparsity.
    #
    #
    A = fill_A_sparse(catf,Lvec,Pvec,Mvec,Lead,Som)
    A_trans = fill_A_trans_sparse(catf,Lvec,Pvec,Mvec,Lead,Som)
    #
    gf = fill_gf_sparse_v2(catf,Lvec,Cvec,Som,Lead) 
    #
    # Initialisation (the result should be independent of initialisation vector)
    #
    p_new = {n: np.random.random()*gf[n] for n in gf}
    p_old = {}
    #
    normdiff = eps+1
    #
    t = 0
    while (t<it) & (normdiff>eps):
        normdiff = 0
        p_old = p_new.copy()
        p_new = {}
        # We search the lines of A which contain non-zero entries coinciding with the non-zero
        # entries of p_old.
        mlines = set()
        for key in p_old:
            for tutu in A_trans[key]:
                mlines.add(tutu)
            #mlines = mlines.union(set(A_trans[key].keys()))
        #print("p_old",p_old)
        for tutu in gf:
            mlines.add(tutu)
        #mlines = mlines.union(set(bi.keys()))
        #print("mlines",mlines)
        for user in mlines:
            p_new[user] = 0
            for leader in Lead[user]:
                if leader in p_old:
                    p_new[user] += A[user][leader]*p_old[leader]
            if user in gf.keys():
                p_new[user]+=gf[user]
            # Norm 1 criterion:
            #normdiff += abs(p_old[user]-p_new[user])
            # Norm INF criterion:
            if user in p_old.keys():
                if abs(p_old[user]-p_new[user])>normdiff:
                    normdiff = abs(p_old[user]-p_new[user])
            else:
                if abs(p_new[user])>normdiff:
                    normdiff = abs(p_new[user])
        t += 1
        #Tracer()()
        #print("p_new",p_new)
    #
    # print("t=",t,"\n")
    # print("diff_last=",normdiff,"\n")
    return p_new


def modelSolve2(N,S,Lambda,Mu,Lead,it=1000,eps=.001,vect=True):
    """ vect if we want to return vectors instead of dicts 
    this time we compute only for S-1 categories and deduce the last one
    we comment Phi and Psi because we actually don't use them"""

    # init
    pNews, qWall = dict(), dict() 
    #Psi, Phi = dict(), dict()
    Cvec,Mvec = Lambda,Mu
    Lvec = {u:Lambda.sum(axis=1)[u] for u in range(N)}
    Som = som_sparse(Lvec,Mvec,Lead)
    Pvec = np.ones((N,S))
    
    # all but one category
    for catf in range(S-1):
        C = fill_C_sparse(catf,Lvec,Pvec,Mvec)
        pNews[catf] = pi_method_sparse_v21(N,catf,Lvec,Pvec,Cvec,Mvec,Lead,Som,it,eps)
        hf = fill_hf_sparse_v2(catf,Lvec,Cvec,Mvec)
        qWall[catf]={}
        #Phi[catf], Psi[catf] = 0, 0
        for userj in pNews[catf]:
            qWall[catf][userj] = C[userj]*pNews[catf][userj]+hf[userj]
            #Psi[catf] += qWall[catf][userj]
            #Phi[catf] += pNews[catf][userj]
        #Psi[catf] = Psi[catf]/N
        #Phi[catf] = Phi[catf]/N

    # last category (if vect we do it below)
    if not vect:
        pNews[S-1] = {n: 1-sum([pNews[s][n] for s in range(S-1)]) for n in range(N)}
        qWall[S-1] = {n: 1-sum([qWall[s][n] for s in range(S-1)]) for n in range(N)}
        #Phi = {s: np.mean([pNews[s][n] for n in range(N)]) for s in range(S)}
        #Psi = {s: np.mean([qWall[s][n] for n in range(N)]) for s in range(S)}
        to_return = (pNews, qWall)#, Phi, Psi)
    
    # vectorise if wanted
    else:
        P, Q = np.zeros((N,S)), np.zeros((N,S))
        for s in range(S-1):
            for n in range(N):
                P[n,s] = pNews[s][n]
                Q[n,s] = qWall[s][n]
        P[:,-1] = 1-P[:,:-1].sum(axis=1)
        Q[:,-1] = 1-Q[:,:-1].sum(axis=1)
        #Phi = P.mean(axis=0)
        #Psi = Q.mean(axis=0)
        to_return = (P, Q)#, Phi, Psi)
    
    # end
    return to_return