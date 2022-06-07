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

def generate_network(uz):
    N = PARAM['num_nodes']
    A_dt = np.eye(N,N)
    for i in range(len(uz)):
        for j in range(i + 1, len(uz)):
            # use the data except the IDs
            zi, zj = np.array(uz.iloc[i,:]), np.array(uz.iloc[j,:])
            # B, C, D, E
            logit = np.exp(PARAM['alpha0'] - PARAM['alpha1'] * np.sqrt(np.square(zi-zj))[0]) * PARAM['network_density']
            # A
            # logit = np.exp(PARAM['alpha0'] - PARAM['alpha1'] * np.sqrt(np.square(zi - np.array(0.5)))[0]) * PARAM['network_density']

            friend = np.random.binomial(1, logit / (1 + logit))
            A_dt[i][j], A_dt[j][i] = friend, friend
    network_density.append(((A_dt.sum()-N)*2/(N*(N-1))))
    return pd.DataFrame(A_dt)

def cal_influence(A, y_binary, uz, uz_pi):
    '''
    calculate the term \sum_j A_ijY_j / \sum_j A_ij
    return: influence: (Num_nodes)，向量元素i代表节点i受到的influence值
    '''
    A = np.array(A)
    A = A - sp.dia_matrix((A.diagonal()[np.newaxis, :], [0]), shape=A.shape) # A-D
    N = len(y_binary) # num of the node
    friend_dict = {}    # dictionary: {'focal_id': [friends' id]}

    for i in range(N):     # construct the t vector for each node
        (col, friend_list) = np.where(A[i]>0)
        friend_dict[str(i)] = friend_list

    # [B] Default: equal influence weight
    influence_weight = np.ones((N, N))  # 矩阵中aij代表i对j的influence (Default = 1)
    influence_weight_observed = np.ones((N, N))

    # [D] Uniform weight
    # for i in range(N):
    #     i_inf = np.array(uz.iloc[i, :]) + 0.5
    #     influence_weight[i,:] = i_inf
    # [E] Uniform weight
    # for i in range(N):
    #     for j in range(i, N):
    #         ij_inf = 1.5 - np.abs(uz.iloc[i,:]-uz.iloc[j,:])
    #         influence_weight[i][j] = ij_inf
    #         influence_weight[j][i] = ij_inf

    A = np.array(A)
    influence_true, influence_observed = np.zeros(N), np.zeros(N)
    for j in range(N):
        denom = len(friend_dict[str(j)])
        if denom==0:
            influence_true[j] = 0
            continue
        numer = sum(y_binary * influence_weight[:, j] * A[:, j])
        numer_observed = sum(y_binary * influence_weight_observed[:, j] * A[:, j])
        influence_true[j] = numer/denom
        influence_observed[j] = numer_observed / denom

    # print(influence_weight)
    return influence_true, influence_observed

def generate_next_Y(y, Influence, Z, set_columns, seed):
    np.random.seed(seed)
    # (1, treat_dim) * (treat_dim, num_nodes) -> (1, num_nodes)
    # Influence = np.matmul(np.array(PARAM['betaT'])[np.newaxis,:], np.array(T).T)
    # (1, featureX_dim) * (featureX_dim, num_nodes) -> (1, num_nodes)
    termZ = np.matmul(np.array(PARAM['betaZ'])[np.newaxis,:], np.array(Z).T)

    Inf = np.array([i*PARAM['betaT'] for i in Influence])
    Inf = Inf[np.newaxis,:].T.astype(float)
    eps = np.random.normal(0, PARAM['epsilon'], size=len(y))[np.newaxis,:].T
    Logit = PARAM['beta0'] + PARAM['beta1'] * y + Inf + termZ.T + eps
    # Logit: np.array(num_nodes, 1)
    # adopt_prob = np.exp(Logit)/(1+np.exp(Logit))
    # y_next = np.random.binomial(1, p=adopt_prob)
    y_next = Logit
    y_next.columns = set_columns
    return y_next

def main():
    # generate the network A
    z = np.random.uniform(0,1,size=PARAM['num_nodes'])
    uz = pd.DataFrame(z, columns=['z'])
    z_pi = np.random.uniform(0,1,size=PARAM['num_nodes'])
    uz_pi = pd.DataFrame(z_pi, columns=['z_pi'])

    A = generate_network(uz)
    A.to_csv(DATA_PATH['Unet'], index=False)
    # generate y0
    # A, B, D, E
    y0 = pd.DataFrame(np.random.normal(PARAM['betaZ']*(z-0.5), 0.2, size=PARAM['num_nodes']), columns=['y0'])  # y(t-1)
    # C
    # y0 = pd.DataFrame(np.random.normal(0, 0.2, size=PARAM['num_nodes']), columns=['y0'])  # y(t-1)
    y0_binary = y0.copy()
    y0_binary[y0_binary<0]=0
    y0_binary[y0_binary>0]=1

    # generate T0
    influence_true, influence_0 = cal_influence(A, np.array(y0_binary).T[0], uz, uz_pi)   # (num_nodes, treat_dim)

    # generate y1
    # y0: (num_nodes, 1), motif_vec_0: (num_nodes, treat_dim), ux0.iloc[:,1:]: (num_nodes, feature_dim)
    y1 = pd.DataFrame(generate_next_Y(y0, influence_true, uz, set_columns=['y1'], seed=seed))
    influence_true = pd.DataFrame(influence_true, columns=['influence_true'])
    influence_0 = pd.DataFrame(influence_0, columns=['influence_0'])

    dt_train = pd.concat([uz, y0, y1, influence_true, influence_0], axis=1)
    dt_train.to_csv(DATA_PATH['save_file'], index=False)


PARAM = {
    # 0. causal graph
    'causal_graph': 'B',

    # 1. net_param
    'alpha0': 0,
    'alpha1': 3,

    # 2. network size and dense
    'network_density': 0.3,
    'num_nodes': 100,

    # 3. network weight
    # describe how the network weights generated by z
    # N-No, U-Uniform
    'weight': 'N',

    # All fixed
    'epsilon': 0.2, # fixed
    'beta0': 0,     # fixed
    'beta1': 1,   # fixed
    'betaT': 1,   # fixed
    'betaZ': [1],  # fixed
}

if __name__ == "__main__":
    data = str(PARAM['causal_graph']) + '_' + str(PARAM['alpha0']) + '_' + str(PARAM['alpha1']) + '_' + str(PARAM['network_density']) + '_' + str(PARAM['num_nodes']) + '_' + str(PARAM['weight'])
    dir_dt = 'data/gendt/' + data
    if not os.path.isdir(dir_dt):
        os.makedirs(dir_dt)
    network_density = []
    for seed in range(11,111):
        set_seed(seed)
        DATA_PATH = {
            'Unet': dir_dt +'/net_' + str(seed) + '.csv',
            'save_file': dir_dt +'/gendt_' + str(seed) + '.csv',
        }
        print("generate network and data:",seed)
        main()
    print("average edge prob:", np.mean(network_density))