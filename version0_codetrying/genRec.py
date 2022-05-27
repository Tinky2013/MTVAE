import pandas as pd
import numpy as np
import scipy.sparse as sp
import random
import math
import seaborn as sns
import matplotlib.pyplot as plt
import os
'''

生成的数据为N*K行
1~K行为第一个User，K+1~2K行为第二个User

'''
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def generate_next_Y(Y0, Y0_binary, Pi):
    K, N = PARAM['num_items'], PARAM['num_nodes']
    Y0_binary = np.array(Y0_binary)
    Y0 = np.array(Y0)
    Inf = np.zeros((N,K))
    for i in range(N):
        for k in range(K):
            observed = random.sample(range(0, N), int(N/10))    # 设置每个人只能观测到十分之一的群体
            Inf[i][k] = (np.sum([Y0_binary[j,k] for j in observed] ))/(int(N/10))

    Y0 = Y0.reshape(-1,1)   # (N*K,1)
    Inf = Inf.reshape(-1,1)
    Pi = Pi.reshape(-1,1)

    eps = np.random.normal(0, PARAM['epsilon'], size=N*K)[np.newaxis,:].T
    Logit = PARAM['beta0'] + PARAM['beta1'] * Y0 + PARAM['betaT'] * Inf + PARAM['betaZ'] * Pi + eps
    # Logit: np.array(num_nodes*num_items,1)
    y_next = Logit
    return y_next, Y0, Inf, Pi

def main():
    K, N = PARAM['num_items'], PARAM['num_nodes']
    # generate item attribute and user attribute
    Zk = np.random.uniform(0,1,size=K)
    Zi = np.random.uniform(0,1,size=N)

    Y0 = np.zeros((N, K))
    Pi = np.zeros((N, K))
    for i in range(PARAM['num_nodes']):
        # Lik: (1,K)
        Lik = Zk * Zi[i]
        Y0[i,:] = np.random.normal(PARAM['betaZ'] * (Zk * Zi[i] - 1/3) , 0.2)
        Pi[i,:] = Zk * Zi[i]

    Y0 = pd.DataFrame(Y0)
    Y0_binary = Y0.copy()
    Y0_binary[Y0_binary<0]=0
    Y0_binary[Y0_binary>0]=1
    y1, flatY0, ave_Inf, ave_Pi = generate_next_Y(Y0, Y0_binary, Pi)


    # Y0: (num_nodes, num_items)
    y1 = pd.DataFrame(y1, columns=['y1'])
    flatY0 = pd.DataFrame(flatY0, columns=['y0'])
    ave_Inf = pd.DataFrame(ave_Inf, columns=['influence'])
    ave_Pi = pd.DataFrame(ave_Pi, columns=['z'])

    dt_train = pd.concat([ave_Pi, flatY0, y1, ave_Inf], axis=1)
    dt_train.to_csv(DATA_PATH['save_file'], index=False)


PARAM = {
    # 0. causal graph
    'causal_graph': 'A',

    # 1. Rec parameters
    'num_nodes': 100,
    'num_items': 30,

    # All fixed
    'epsilon': 0.2, # fixed
    'beta0': 0,     # fixed
    'beta1': 0.7,   # fixed
    'betaT': 0.5,   # fixed
    'betaZ': [0.3],  # fixed
}

if __name__ == "__main__":
    data = str(PARAM['causal_graph']) + '_' + str(PARAM['num_nodes']) + '_' + str(PARAM['num_items'])
    dir_dt = 'data/genRec/' + data
    if not os.path.isdir(dir_dt):
        os.makedirs(dir_dt)
    for seed in range(11,111):
        set_seed(seed)
        DATA_PATH = {
            'save_file': dir_dt +'/genRec_' + str(seed) + '.csv',
        }
        print("generate rec data:",seed)
        main()