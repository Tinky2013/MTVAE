import numpy as np
import pandas as pd
import scipy.sparse as sp

'''
input dataset
Graph: {0,1} Adj Matrix     -- Unet
outcome: Y(t-1)             -- y
user profile                -- Uprofile

output dataset: final dataset for training  -- gendt
[profile, treatment, Y(t-1)]
'''

def cal_motif_vec(PATH):
    y_orig = pd.read_csv(PATH['y'])
    A = np.array(pd.read_csv(PATH['Unet']))
    y = np.array(y_orig)
    profile = pd.read_csv(PATH['Uprofile'])
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
    dt = pd.concat([profile, motif, y_orig], axis=1)
    dt.to_csv(PATH['save_file'], index=False)

PATH_train = {
    'Unet': 'data/Unet_train.csv',
    'y': 'data/y_train.csv',
    'Uprofile': 'data/Uprofile_train.csv',
    'save_file': 'data/gendt_train.csv',
}
PATH_test = {
    'Unet': 'data/Unet_test.csv',
    'y': 'data/y_test.csv',
    'Uprofile': 'data/Uprofile_test.csv',
    'save_file': 'data/gendt_test.csv',
}

cal_motif_vec(PATH_train)
cal_motif_vec(PATH_test)


