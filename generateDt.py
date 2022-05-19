import pandas as pd
import numpy as np
import scipy.sparse as sp
import random
import math
'''

根据用户的相似性生成用户网络

'''
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def generate_network(ux, uz):
    A_dt = np.eye(len(ux), len(ux))
    edge_prob = []
    for i in range(len(ux)):
        for j in range(i + 1, len(ux)):
            # use the data except the IDs
            xi, xj = np.array(ux.iloc[i, 1:]), np.array(ux.iloc[j, 1:])
            zi, zj = np.array(uz.iloc[i, 1:]), np.array(uz.iloc[j, 1:])
            logit = np.exp(PARAM['alpha0'] - PARAM['alpha1'] * np.linalg.norm([xi, xj]) - PARAM['alpha2'] * np.linalg.norm([zi, zj]))
            logit = logit * PARAM['network_density']
            friend = np.random.binomial(1, logit / (1 + logit))
            edge_prob.append(logit)
            A_dt[i][j], A_dt[j][i] = friend, friend
        if i%10 == 0:
            print("generate: ",i," nodes")
    print("Mean edge prob:", np.mean(edge_prob))
    return pd.DataFrame(A_dt)

def dealMinMax(vec):
    Min = min(vec)
    Max = max(vec)
    return (vec-min(vec))/(max(vec)-min(vec)+1e-3) + 1e-6

def cal_motif_vec(PATH, y):
    '''
    y: account for the action of neighbor nodes
    '''
    A = np.array(pd.read_csv(PATH['Unet']))
    A = A - sp.dia_matrix((A.diagonal()[np.newaxis, :], [0]), shape=A.shape)    # A-D
    N = len(y) # num of the node
    friend_dict = {}    # dictionary: {'focal_id': [friends' id]}
    motif = np.zeros((N, 8))
    # this is for synthetic data (different influential weight)
    influence_weight = np.zeros((N,N)) # 矩阵中aij代表i对j的influence (Default = 1)
    for i in range(N):     # construct the t vector for each node
        (col, friend_list) = np.where(A[i]>0)
        friend_dict[str(i)] = friend_list

    for i in range(N):
        Neighbor = friend_dict[str(i)]
        J1 = len(Neighbor)
        # J1 is the number of neighbors
        if J1==0:   # Node i has 0 neighbor（没有邻居时，这个节点对任何其他节点的影响aij都为0）
            motif[i] = [0]*8
            continue
        elif J1==1: # Node i has 1 neighbor
            Ni = Neighbor[0]    # Node i's neighbor's Id
            influence_weight[i][Ni] = 1 # 一个邻居时，节点i对其邻居Ni的影响为aij为1
            influence_weight[Ni][i] = 1  # 只有一个朋友，收到的影响也为1
            if y[Ni] == 0:
                motif[i] = [0, 1, 0, 0, 0, 0, 0, 0]
            else:
                motif[i] = [1, 0, 0, 0, 0, 0, 0, 0]
            continue

        else:
            # 在节点i的朋友中，判断他们朋友之间的关系来决定权重
            # 对于邻居节点而言的(same_friend_num, different_friend_num, not_friend_num)
            friend_type_count = np.zeros((J1,3))
            for j in range(J1):
                iN1 = Neighbor[j]
                for k in range(j+1, J1):
                    iN2 = Neighbor[k]
                    if (iN1 in friend_dict[str(iN2)]):    # they are friends
                        if y[iN1] == y[iN2]:
                            friend_type_count[j][0] += 1
                            friend_type_count[k][0] += 1
                        else:
                            friend_type_count[j][1] += 1
                            friend_type_count[k][1] += 1
                    else:
                        friend_type_count[j][2] += 1
                        friend_type_count[k][2] += 1
            # friend_type_count记录了节点i的邻居（编号为Neighbor[j]）各种（和i相连的）朋友的数量
            for j in range(J1):
                # 这个邻居对focal node i的影响权重
                influence_weight[Neighbor[j]][i] = (friend_type_count[j][0] * PARAM['same_inf'] + friend_type_count[j][1] * PARAM['diff_inf']+ friend_type_count[j][2] * PARAM['none_inf']) / J1

            m0, m1, m2, m3, m4, m5, m6, m7 = 0, 0, 0, 0, 0, 0, 0, 0
            for j in range(J1):
                iN1 = Neighbor[j]
                if y[iN1] == 1:
                    m0 += 1
                else:
                    m1 += 1
                for k in range(j + 1, J1):
                    iN2 = Neighbor[k] # iN1, iN2 are friends' id
                    # judge whether friend_dict[str(i)][j1] and friend_dict[str(i)][j2] are friends
                    if (iN1 in friend_dict[str(iN2)]):    # they are friends
                        if (y[iN1]==1 and y[iN2]==1):
                            m5 += 1
                        elif (y[iN1] == 0 and y[iN2] == 0):
                            m7 += 1
                        else:
                            m6 += 1
                    else:   # they are not friends
                        if (y[iN1]==1 and y[iN2]==1):
                            m2 += 1
                        elif (y[iN1] == 0 and y[iN2] == 0):
                            m4 += 1
                        else:
                            m3 += 1
            JS, JC, = m2 + m3 + m4, m5 + m6 + m7
            if JS==0 or JC==0:
                motif[i][2] = m2 / max(JS, JC)
                motif[i][3] = m3 / max(JS, JC)
                motif[i][4] = m4 / max(JS, JC)
                motif[i][5] = m5 / max(JS, JC)
                motif[i][6] = m6 / max(JS, JC)
                motif[i][7] = m7 / max(JS, JC)

            else:
                motif[i][2] = m2 / JS
                motif[i][3] = m3 / JS
                motif[i][4] = m4 / JS
                motif[i][5] = m5 / JC
                motif[i][6] = m6 / JC
                motif[i][7] = m7 / JC

            motif[i][0] = m0 / J1
            motif[i][1] = m1 / J1

    motif = pd.DataFrame(motif, columns=['t0','t1','t2','t3','t4','t5','t6','t7'])
    for i in motif.columns:
        motif[i] = dealMinMax(motif[i])
    # calculate weighted influence (\sum_j A_ij Y_ij / \sum_j A_ij)
    influence = np.zeros(N)
    for j in range(N):
        denom = sum(influence_weight[:,j])
        numer = sum(y*influence_weight[:,j])
        influence[j] = denom/numer
    return motif, influence

def generate_next_Y(y, Influence, X, Z):
    # (1, treat_dim) * (treat_dim, num_nodes) -> (1, num_nodes)
    # Influence = np.matmul(np.array(PARAM['betaT'])[np.newaxis,:], np.array(T).T)
    # (1, featureX_dim) * (featureX_dim, num_nodes) -> (1, num_nodes)
    termX = np.matmul(np.array(PARAM['betaX'])[np.newaxis,:], np.array(X.iloc[:, 1:]).T)
    termZ = np.matmul(np.array(PARAM['betaZ'])[np.newaxis,:], np.array(Z.iloc[:, 1:]).T)

    Inf = np.array([i*PARAM['betaT'] for i in Influence])
    Inf = Inf[np.newaxis,:].T.astype(float)

    Logit = PARAM['beta0'] + PARAM['beta1'] * y + Inf + termX.T + termZ.T + np.random.normal(0, PARAM['epsilon'])
    # Logit: np.array(num_nodes, 1)
    adopt_prob = np.exp(Logit)/(1+np.exp(Logit))
    y_next = np.random.binomial(1, p=adopt_prob)
    return y_next

def main():
    # generate the network A
    ux_train = pd.read_csv(PATH_train['Ux']) # observed features
    uz_train = pd.read_csv(PATH_train['Uz']) # unobserved features
    N_train = len(ux_train)
    A_train = generate_network(ux_train, uz_train)
    A_train.to_csv(PATH_train['Unet'], index=False)

    # ux_test = pd.read_csv(PATH_test['Ux'])
    # uz_test = pd.read_csv(PATH_test['Uz'])
    # N_test = len(ux_test)
    # A_test = generate_network(ux_test, uz_test)
    # A_test.to_csv(PATH_test['Unet'], index=False)

    # generate y0
    y0_train = pd.DataFrame(np.random.binomial(1, p=PARAM['p0'], size=N_train), columns=['y0'])  # y(t-1)
    # generate T
    motif_vec_train, influence = cal_motif_vec(PATH_train, np.array(y0_train).T[0])   # (num_nodes, treat_dim)
    # generate y1
    # y0: (num_nodes, 1), motif_vec_train: (num_nodes, treat_dim), ux_train.iloc[:,1:]: (num_nodes, feature_dim)
    y1_train = pd.DataFrame(generate_next_Y(y0_train, influence, ux_train, uz_train), columns=['y1'])
    dt_train = pd.concat([ux_train, uz_train.drop(labels=['Id'], axis=1, inplace=True), motif_vec_train, y0_train, y1_train], axis=1)
    dt_train.to_csv(PATH_train['save_file'], index=False)

    # y0_test = pd.DataFrame(np.random.binomial(1, p=PARAM['p0'],  size=N_test), columns=['y0'])  # y(t-1)
    # motif_vec_test, influence = cal_motif_vec(PATH_test, np.array(y0_test).T[0])
    # y1_test = pd.DataFrame(generate_next_Y(y0_test, influence, ux_test, uz_test), columns=['y1'])
    # dt_test = pd.concat([ux_test, uz_test.drop(labels=['Id'], axis=1, inplace=True), motif_vec_test, y0_test, y1_test],
    #                     axis=1)
    # dt_test.to_csv(PATH_test['save_file'], index=False)


PATH_train = {
    'Unet': 'data/Unet_train.csv',
    'Ux': 'data/Ux_train.csv',
    'Uz': 'data/Uz_train.csv',
    'save_file': 'data/gendt_train.csv',
}
PATH_test = {
    'Unet': 'data/Unet_test.csv',
    'Ux': 'data/Ux_test.csv',
    'Uz': 'data/Uz_test.csv',
    'save_file': 'data/gendt_test.csv',
}
PARAM = {
    'p0': 0.5,
    'epsilon': 1,
    'alpha0': 1,
    'alpha1': 1,
    'alpha2': 1,
    'network_density': 3,

    'beta0': 0,
    'beta1': 1,
    'betaT': 0.3,
    'same_inf': 1.2,
    'diff_inf': 0.8,
    'none_inf': 1,

    'betaX': [0.2,0.25,0.05,-0.2,-0.05],
    'betaZ': [0.2,0.15,0.05,-0.05,-0.15],

}

if __name__ == "__main__":
    set_seed(100)
    main()