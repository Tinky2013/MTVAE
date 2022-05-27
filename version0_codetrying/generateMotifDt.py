import pandas as pd
import numpy as np
import scipy.sparse as sp
import random
import math
import seaborn as sns
import matplotlib.pyplot as plt
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
        # if i%10 == 0:
        #     print("generate: ",i," nodes")
    print("Mean edge prob:", np.mean(edge_prob))
    return pd.DataFrame(A_dt)

def dealMinMax(vec):
    Min = min(vec)
    Max = max(vec)
    return (vec-min(vec))/(max(vec)-min(vec)+1e-3) + 1e-6

def cal_motif_vec(A, y_binary, set_columns):
    '''
    y: account for the action of neighbor nodes
    '''
    A = np.array(A)
    A = A - sp.dia_matrix((A.diagonal()[np.newaxis, :], [0]), shape=A.shape)    # A-D
    N = len(y_binary) # num of the node
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
            if y_binary[Ni] == 0:
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
                        if y_binary[iN1] == y_binary[iN2]:
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
                influence_weight[Neighbor[j]][i] = (friend_type_count[j][0] * PARAM['same_inf'] + friend_type_count[j][1] * PARAM['diff_inf']+ friend_type_count[j][2] * PARAM['none_inf']) /   (J1-1)

            m0, m1, m2, m3, m4, m5, m6, m7 = 0, 0, 0, 0, 0, 0, 0, 0
            for j in range(J1):
                iN1 = Neighbor[j]
                if y_binary[iN1] == 1:
                    m0 += 1
                else:
                    m1 += 1
                for k in range(j + 1, J1):
                    iN2 = Neighbor[k] # iN1, iN2 are friends' id
                    # judge whether friend_dict[str(i)][j1] and friend_dict[str(i)][j2] are friends
                    if (iN1 in friend_dict[str(iN2)]):    # they are friends
                        if (y_binary[iN1]==1 and y_binary[iN2]==1):
                            m5 += 1
                        elif (y_binary[iN1] == 0 and y_binary[iN2] == 0):
                            m7 += 1
                        else:
                            m6 += 1
                    else:   # they are not friends
                        if (y_binary[iN1]==1 and y_binary[iN2]==1):
                            m2 += 1
                        elif (y_binary[iN1] == 0 and y_binary[iN2] == 0):
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

    motif = pd.DataFrame(motif, columns=set_columns)
    for i in motif.columns:
        motif[i] = dealMinMax(motif[i])
    # calculate weighted influence (\sum_j A_ij Y_ij / \sum_j A_ij)
    influence = np.zeros(N)

    for j in range(N):
        denom = len(friend_dict[str(j)])
        if denom==0:
            influence[j] = 0
            continue
        numer = sum(y_binary * influence_weight[:, j])
        influence[j] = numer/denom

    return motif, influence

def generate_next_Y(y, Influence, X, Z, set_columns, seed):
    np.random.seed(seed)
    # (1, treat_dim) * (treat_dim, num_nodes) -> (1, num_nodes)
    # Influence = np.matmul(np.array(PARAM['betaT'])[np.newaxis,:], np.array(T).T)
    # (1, featureX_dim) * (featureX_dim, num_nodes) -> (1, num_nodes)
    termX = np.matmul(np.array(PARAM['betaX'])[np.newaxis,:], np.array(X.iloc[:, 1:]).T)
    termZ = np.matmul(np.array(PARAM['betaZ'])[np.newaxis,:], np.array(Z.iloc[:, 1:]).T)

    Inf = np.array([i*PARAM['betaT'] for i in Influence])
    Inf = Inf[np.newaxis,:].T.astype(float)
    eps = np.random.normal(0, PARAM['epsilon'], size=len(y))[np.newaxis,:].T

    # check the distributions
    # sns.distplot(y,kde=True)
    # plt.show()
    # sns.distplot(Inf,kde=True)
    # plt.show()
    # sns.distplot(termX,kde=True)
    # plt.show()
    # sns.distplot(termZ,kde=True)
    # plt.show()
    # sns.displot(eps,kde=True)
    # plt.show()


    Logit = PARAM['beta0'] + PARAM['beta1'] * y + Inf + termX.T + termZ.T + eps
    # Logit: np.array(num_nodes, 1)
    # adopt_prob = np.exp(Logit)/(1+np.exp(Logit))
    # y_next = np.random.binomial(1, p=adopt_prob)
    y_next = Logit
    y_next.columns = set_columns
    return y_next

def main():
    # generate the network A
    ux = pd.read_csv(DATA_PATH['Ux']) # observed features
    uz = pd.read_csv(DATA_PATH['Uz']) # unobserved features
    N_nodes = len(ux)
    A = generate_network(ux, uz)
    A.to_csv(DATA_PATH['Unet'], index=False)


    # generate y0
    y0 = pd.DataFrame(np.random.normal(0, 0.5, size=N_nodes), columns=['y0'])  # y(t-1)
    y0_binary = y0.copy()
    y0_binary[y0_binary<0]=0
    y0_binary[y0_binary>0]=1

    # generate T0
    motif_vec_0, influence_0 = cal_motif_vec(A, np.array(y0_binary).T[0], set_columns=['t00','t01','t02','t03','t04','t05','t06','t07'])   # (num_nodes, treat_dim)

    # generate y1
    # y0: (num_nodes, 1), motif_vec_0: (num_nodes, treat_dim), ux0.iloc[:,1:]: (num_nodes, feature_dim)
    y1 = pd.DataFrame(generate_next_Y(y0, influence_0, ux, uz, set_columns=['y1'], seed=seed))
    influence_0 = pd.DataFrame(influence_0, columns=['influence_0'])

    # generate T1
    y1_binary = y1.copy()
    y1_binary[y1_binary<0]=0
    y1_binary[y1_binary>0]=1
    motif_vec_1, influence_1 = cal_motif_vec(A, np.array(y1_binary).T[0], set_columns=['t10','t11','t12','t13','t14','t15','t16','t17'])   # (num_nodes, treat_dim)
    # generate y2
    y2 = pd.DataFrame(generate_next_Y(y1, influence_1, ux, uz, set_columns=['y2'], seed=seed+1))
    influence_1 = pd.DataFrame(influence_1, columns=['influence_1'])


    dt_train = pd.concat([ux, uz.drop(labels=['Id'], axis=1, inplace=True), motif_vec_0, motif_vec_1, y0, y1, y2, influence_0, influence_1], axis=1)
    dt_train.to_csv(DATA_PATH['save_file'], index=False)

if __name__ == "__main__":
    PARAM = {
        'p0': 0.5,
        'epsilon': 0.05,    # fixed
        # net_param
        'alpha0': 0,
        'alpha1': 3,
        'alpha2': 3,
        'network_density': 2,

        'beta0': -0.5,  # -0.5*(betaT+betaZ)
        'beta1': 0.6,   # fixed
        'betaT': 0.8,   # fixed
        'same_inf': 2.5,
        'diff_inf': 0,
        'none_inf': 0.5,

        'betaX': [0.5], # fixed
        'betaZ': [0.2], # fixed
    }
    for seed in range(11,111):
        # graph = 'net_0_3_3_3_inf_2.8_0_0.2'
        graph = 'test'
        set_seed(seed)
        DATA_PATH = {
            'Unet': 'data/network/'+graph+'/Unet_' + str(seed) + '.csv',
            'Ux': 'data/Ux_train.csv',
            'Uz': 'data/Uz_train.csv',
            'save_file': 'data/gendt/'+graph+'/gendt_' + str(seed) + '.csv',
        }
        print("generate network and data:",seed)
        main()