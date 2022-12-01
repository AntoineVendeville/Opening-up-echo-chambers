#!/usr/bin/env python
# coding: utf-8

import sys
from optim_ineq_fct import maxGini_ineq
import numpy as np
import pickle 

# sys argv
in_path = sys.argv[1]
out_path = sys.argv[2]
B = float(sys.argv[3])
gamma1 = float(sys.argv[4])
gamma2 = float(sys.argv[5])
penalty = bool(int(sys.argv[6]))
p_init_ = sys.argv[7] # 'estim' or 'theo'
numfocus = int(sys.argv[8])

# Load data.
p_init = pickle.load(open(in_path+f'p_{p_init_}_matrix.p','rb'))
Lambda = pickle.load(open(in_path+'Lambda_matrix.p','rb'))
Mu = pickle.load(open(in_path+'Mu_matrix.p','rb'))
Leaders = pickle.load(open(in_path+'Leaders.p','rb'))
N, S = Lambda.shape

# run and save
Brange = [.5] #(.02, .04, .06, .08, .1, .2)
for B in Brange:
    print(f'\nB={B}\n')
    nu, p, runtime = maxGini_ineq(N,S,Lambda,Mu,Leaders,B,gamma1=gamma1,gamma2=gamma2, penalty=penalty, verbose=True, p_init=p_init,numfocus=numfocus)
    pickle.dump(nu, open(out_path+f'nu_{B}.p','wb'))
    pickle.dump(p, open(out_path+f'p_{B}.p','wb'))
    pickle.dump(runtime, open(out_path+f'runtime_{B}.p','wb'))
    if penalty: 
        pickle.dump(gamma1, open(out_path+f'gamma1_{B}.p', 'wb'))
