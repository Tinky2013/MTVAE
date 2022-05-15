import pandas as pd
import numpy as np
import scipy.sparse as sp
import random
'''

根据用户的相似性生成用户网络

'''
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def generate_network(ux, uz):
    A_dt = np.eye(len(ux), len(ux))
    for i in range(len(ux)):
        for j in range(i + 1, len(ux)):
            # use the data except the IDs
            xi, xj = np.array(ux.iloc[i, 1:]), np.array(ux.iloc[j, 1:])
            zi, zj = np.array(uz.iloc[i, 1:]), np.array(uz.iloc[j, 1:])
            logit = np.exp(PARAM['alpha0'] - PARAM['alpha1'] * np.linalg.norm([xi, xj]) - PARAM['alpha2'] * np.linalg.norm([zi, zj]))
            friend = np.random.binomial(1, logit / (1 + logit))
            A_dt[i][j], A_dt[j][i] = friend, friend
    return pd.DataFrame(A_dt)

def dealMinMax(vec):
    Min = min(vec)
    Max = max(vec)
    return (vec-min(vec))/(max(vec)-min(vec)+1e-3) + 1e-6

def cal_motif_vec(PATH, y):
    A = np.array(pd.read_csv(PATH['Unet']))
    A = A - sp.dia_matrix((A.diagonal()[np.newaxis, :], [0]), shape=A.shape)    # A-D
    N = len(y) # num of the node
    friend_dict = {}    # dictionary: {'focal_id': [friends' id]}
    motif = np.zeros((N, 8))
    for i in range(N):     # construct the t vector for each node
        (col, friend_list) = np.where(A[i]>0)
        friend_dict[str(i)] = friend_list

    for i in range(N):
        Neighbor = friend_dict[str(i)]
        J1, J2 = len(Neighbor), len(Neighbor) * (len(Neighbor) - 1)
        # J1 is the number of neighbors
        if J1==0:
            motif[i] = [0]*8
            continue
        if J1==1:
            if y[Neighbor[0]] == 0:
                motif[i] = [0, 1, 0, 0, 0, 0, 0, 0]
            else:
                motif[i] = [1, 0, 0, 0, 0, 0, 0, 0]
            continue

        motif[i][0] = J1 / (N - 1)
        motif[i][1] = (N - 1 - J1) / (N - 1)
        m2, m3, m4, m5, m6, m7 = 0, 0, 0, 0, 0, 0
        for j in range(len(Neighbor)):
            for k in range(j + 1, len(Neighbor)):
                iN1, iN2 = Neighbor[j], Neighbor[k] # iN1, iN2 are friends' id
               # judge whether friend_dict[str(i)][j1] and friend_dict[str(i)][j2] are friends
                if (iN1 in friend_dict[str(iN2)]):    # they are friends
                    if (y[iN1]==1 and y[iN2]==1):
                        m5 = m5 + 1
                    elif (y[iN1] == 0 and y[iN2] == 0):
                        m7+=1
                    else:
                        m6+=1
                else:   # they are not friends
                    if (y[iN1]==1 and y[iN2]==1):
                        m2 = m2 + 1
                    elif (y[iN1] == 0 and y[iN2] == 0):
                        m4+=1
                    else:
                        m3+=1

        motif[i][2] = m2 / J2
        motif[i][3] = m3 / J2
        motif[i][4] = m4 / J2
        motif[i][5] = m5 / J2
        motif[i][6] = m6 / J2
        motif[i][7] = m7 / J2

    motif = pd.DataFrame(motif, columns=['t0','t1','t2','t3','t4','t5','t6','t7'])
    for i in motif.columns:
        motif[i] = dealMinMax(motif[i])
    return motif

def generate_next_Y(y, T, X, Z):
    # (1, treat_dim) * (treat_dim, num_nodes) -> (1, num_nodes)
    Influence = np.matmul(np.array(PARAM['betaT'])[np.newaxis,:], np.array(T).T)
    # (1, featureX_dim) * (featureX_dim, num_nodes) -> (1, num_nodes)
    termX = np.matmul(np.array(PARAM['betaX'])[np.newaxis,:], np.array(X.iloc[:, 1:]).T)
    termZ = np.matmul(np.array(PARAM['betaZ'])[np.newaxis,:], np.array(Z.iloc[:, 1:]).T)
    # Logit: np.array(num_nodes, 1)
    Logit = PARAM['beta0'] + PARAM['beta1'] * np.array(y) + Influence.T + termX.T + termZ.T + np.random.normal(0,1)
    y_next = np.random.binomial(1, p=1/(1+np.exp(-Logit)))
    return y_next

def main():
    # generate the network A
    ux_train, ux_test = pd.read_csv(PATH_train['Ux']), pd.read_csv(PATH_test['Ux']) # observed features
    uz_train, uz_test = pd.read_csv(PATH_train['Uz']), pd.read_csv(PATH_test['Uz']) # unobserved features
    N_train, N_test = len(ux_train), len(ux_test)
    A_train, A_test = generate_network(ux_train, uz_train), generate_network(ux_train, uz_train)
    A_train.to_csv(PATH_train['Unet'], index=False)
    A_test.to_csv(PATH_test['Unet'], index=False)

    # generate y0
    y0_train = pd.DataFrame(np.random.binomial(1, p=PARAM['p0'], size=N_train), columns=['y0'])  # y(t-1)
    y0_test = pd.DataFrame(np.random.binomial(1, p=PARAM['p0'],  size=N_test), columns=['y0'])  # y(t-1)
    # generate T
    motif_vec_train = cal_motif_vec(PATH_train, np.array(y0_train).T[0])   # (num_nodes, treat_dim)
    motif_vec_test = cal_motif_vec(PATH_test, np.array(y0_test).T[0])

    # generate y1
    # y0: (num_nodes, 1), motif_vec_train: (num_nodes, treat_dim), ux_train.iloc[:,1:]: (num_nodes, feature_dim)
    y1_train = pd.DataFrame(generate_next_Y(y0_train, motif_vec_train, ux_train, uz_train), columns=['y1'])
    y1_test = pd.DataFrame(generate_next_Y(y0_test, motif_vec_test, ux_test, uz_test), columns=['y1'])

    dt_train = pd.concat([ux_train, uz_train.drop(labels=['Id'], axis=1, inplace=True), motif_vec_train, y0_train, y1_train], axis=1)
    dt_test = pd.concat([ux_test, uz_test.drop(labels=['Id'], axis=1, inplace=True), motif_vec_test, y0_test, y1_test], axis=1)

    dt_train.to_csv(PATH_train['save_file'], index=False)
    dt_test.to_csv(PATH_test['save_file'], index=False)


PATH_train = {
    'Unet': 'data/Unet_train.csv',
    'y': 'data/y_train.csv',
    'Ux': 'data/Ux_train.csv',
    'Uz': 'data/Uz_train.csv',
    'save_file': 'data/gendt_train.csv',
}
PATH_test = {
    'Unet': 'data/Unet_test.csv',
    'y': 'data/y_test.csv',
    'Ux': 'data/Ux_test.csv',
    'Uz': 'data/Uz_test.csv',
    'save_file': 'data/gendt_test.csv',
}
PARAM = {
    'p0': 0.5,
    'alpha0': 1,
    'alpha1': 1,
    'alpha2': 1,
    'beta0': 0,
    'beta1': 0.7,
    'betaT': [0.3,-0.1,0.45,0.05,-0.2,0.4,0.1,-0.15],
    'betaX': [0.2,0.25,0.05,-0.1,-0.05],
    'betaZ': [0.25,0.15,0.05,-0.1,-0.2]
}

if __name__ == "__main__":
    set_seed(100)
    main()