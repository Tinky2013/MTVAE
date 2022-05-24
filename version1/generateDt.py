import pandas as pd
import numpy as np
import scipy.sparse as sp
import random
import math
import seaborn as sns
import matplotlib.pyplot as plt
import os
'''

根据用户的相似性生成用户网络

'''
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def generate_network(ux, uz):
    N = PARAM['num_nodes']
    A_dt = np.eye(N,N)
    for i in range(len(ux)):
        for j in range(i + 1, len(ux)):
            # use the data except the IDs
            xi, xj = np.array(ux.iloc[i,:]), np.array(ux.iloc[j,:])
            zi, zj = np.array(uz.iloc[i,:]), np.array(uz.iloc[j,:])
            logit = np.exp(PARAM['alpha0'] - PARAM['alpha1'] * np.linalg.norm([xi, xj]) - PARAM['alpha2'] * np.linalg.norm([zi, zj])) * PARAM['network_density']
            friend = np.random.binomial(1, logit / (1 + logit))
            A_dt[i][j], A_dt[j][i] = friend, friend
    network_density.append(((A_dt.sum()-N)*2/(N*(N-1))))
    return pd.DataFrame(A_dt)

def cal_influence(A, y_binary):
    '''
    calculate the term \sum_j A_ijY_j / \sum_j A_ij
    return: influence: (Num_nodes)，向量元素i代表节点i受到的influence值
    '''
    A = np.array(A)
    A = A - sp.dia_matrix((A.diagonal()[np.newaxis, :], [0]), shape=A.shape)    # A-D
    N = len(y_binary) # num of the node
    friend_dict = {}    # dictionary: {'focal_id': [friends' id]}

    for i in range(N):     # construct the t vector for each node
        (col, friend_list) = np.where(A[i]>0)
        friend_dict[str(i)] = friend_list

    influence_weight = np.ones((N, N))  # 矩阵中aij代表i对j的influence (Default = 1)
    influence = np.zeros(N)

    for j in range(N):
        denom = len(friend_dict[str(j)])
        if denom==0:
            influence[j] = 0
            continue
        numer = sum(y_binary * influence_weight[:, j])
        influence[j] = numer/denom

    return influence

def generate_next_Y(y, Influence, X, Z, set_columns, seed):
    np.random.seed(seed)
    # (1, treat_dim) * (treat_dim, num_nodes) -> (1, num_nodes)
    # Influence = np.matmul(np.array(PARAM['betaT'])[np.newaxis,:], np.array(T).T)
    # (1, featureX_dim) * (featureX_dim, num_nodes) -> (1, num_nodes)
    termX = np.matmul(np.array(PARAM['betaX'])[np.newaxis,:], np.array(X).T)
    termZ = np.matmul(np.array(PARAM['betaZ'])[np.newaxis,:], np.array(Z).T)

    Inf = np.array([i*PARAM['betaT'] for i in Influence])
    Inf = Inf[np.newaxis,:].T.astype(float)
    eps = np.random.normal(0, PARAM['epsilon'], size=len(y))[np.newaxis,:].T
    Logit = PARAM['beta0'] + PARAM['beta1'] * y + Inf + termX.T + termZ.T + eps
    # Logit: np.array(num_nodes, 1)
    # adopt_prob = np.exp(Logit)/(1+np.exp(Logit))
    # y_next = np.random.binomial(1, p=adopt_prob)
    y_next = Logit
    y_next.columns = set_columns
    return y_next

def main():
    # generate the network A
    U = np.random.uniform(-0.5,0.5,size=PARAM['num_nodes'])
    z = np.random.uniform(0,1,size=PARAM['num_nodes'])
    x = PARAM['tau1']*U+PARAM['tau2']*(z-0.5)+0.5
    corr.append(np.corrcoef(x,z)[0][1])

    uz = pd.DataFrame(z, columns=['z'])
    ux = pd.DataFrame(x, columns=['x'])

    N_nodes = len(ux)
    A = generate_network(ux, uz)
    A.to_csv(DATA_PATH['Unet'], index=False)

    # generate y0
    y0 = pd.DataFrame(np.random.normal(0, 0.5, size=N_nodes), columns=['y0'])  # y(t-1)
    y0_binary = y0.copy()
    y0_binary[y0_binary<0]=0
    y0_binary[y0_binary>0]=1

    # generate T0
    influence_0 = cal_influence(A, np.array(y0_binary).T[0])   # (num_nodes, treat_dim)

    # generate y1
    # y0: (num_nodes, 1), motif_vec_0: (num_nodes, treat_dim), ux0.iloc[:,1:]: (num_nodes, feature_dim)
    y1 = pd.DataFrame(generate_next_Y(y0, influence_0, ux, uz, set_columns=['y1'], seed=seed))
    influence_0 = pd.DataFrame(influence_0, columns=['influence_0'])

    dt_train = pd.concat([ux, uz, y0, y1, influence_0], axis=1)
    dt_train.to_csv(DATA_PATH['save_file'], index=False)


PARAM = {
    # 1. net_param
    'alpha0': 0,
    'alpha1': 0,
    'alpha2': 3,

    # 2. network size and dense
    'network_density': 3,
    'num_nodes': 100,

    # 3. confounders
    'betaX': [0],  # fixed
    'betaZ': [0.4],  # fixed

    # 4. correlation of confounders
    # tau1=1 and tau2=0 means no correlation
    'tau1': 1,
    'tau2': 0,

    # All fixed
    'epsilon': 0.5,    # fixed
    'beta0': 0,     # fixed
    'beta1': 0.7,   # fixed
    'betaT': 0.3,   # fixed
}

if __name__ == "__main__":
    if PARAM['tau1']==1 and PARAM['tau2']==0:
        graph = str(PARAM['alpha0']) + '_' + str(PARAM['alpha1']) + '_' + str(PARAM['alpha2']) + '_' + str(
            PARAM['network_density']) + '_X' + str(PARAM['betaX']) + '_Z' + str(PARAM['betaZ'])
    else:
        graph = str(PARAM['alpha0']) + '_' + str(PARAM['alpha1']) + '_' + str(PARAM['alpha2']) + '_' + str(
            PARAM['network_density']) + '_X' + str(PARAM['betaX']) + '_Z' + str(PARAM['betaZ']) + '_tau[' + str(PARAM['tau1']) + '_' + str(PARAM['tau2']) + ']'
    print(graph)
    dir = 'data/gendt/'+graph
    network_density = []
    corr = []
    graph = 'test'  ## TODO: testing
    if not os.path.isdir(dir):
        os.makedirs(dir)
    for seed in range(11,21):
        set_seed(seed)
        DATA_PATH = {
            'Unet': dir+'/net_' + str(seed) + '.csv',
            'save_file': dir+'/gendt_' + str(seed) + '.csv',
        }
        print("generate network and data:",seed)
        main()
    print("average edge prob:", np.mean(network_density), "average x,z corr:", np.mean(corr))