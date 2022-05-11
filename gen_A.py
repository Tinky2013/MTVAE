import pandas as pd
import numpy as np

'''

根据用户的相似性生成用户网络

'''

def generate(df):
    x1 = df['gender']
    x2_0 = df['cluster_0']
    x2_1 = df['cluster_1']
    x2_2 = df['cluster_2']
    x3 = df['age']

    t0, t1, t2, t3, t4, t5, t6, t7, y = [], [], [], [], [], [], [], [], []
    for i in range(len(df)):
        t0.append(5 * x1[i] + 5 * x2_0[i] + 0.1 * x3[i] + np.random.normal(0,1))
        t1.append(5 * x1[i] + 5 * x2_1[i] + 0.1 * x3[i] + np.random.normal(0,1))
        t2.append(5 * x1[i] + 5 * x2_2[i] + 0.1 * x3[i] + np.random.normal(0,1))
        t3.append(-5 * x1[i] + 5 * x2_0[i] + 0.1 * x3[i] + np.random.normal(0,1))
        t4.append(-5 * x1[i] + 5 * x2_1[i] + 0.1 * x3[i] + np.random.normal(0,1))
        t5.append(-5 * x1[i] + 5 * x2_2[i] + 0.1 * x3[i] + np.random.normal(0,1))
        t6.append(5 * x1[i] + 0.1 * x3[i] + np.random.normal(0,1))
        t7.append(-5 * x1[i] + 0.005 * x3[i] * x3[i] + np.random.normal(0,1))

    # normalize and avoid exact 0 or 1
    t0, t1, t2, t3, t4, t5, t6, t7 = mx(t0), mx(t1), mx(t2), mx(t3), mx(t4), mx(t5), mx(t6), mx(t7)
    print(min(t0), min(t1), min(t2), min(t3), min(t4), min(t5), min(t6), min(t7))
    print(max(t0),max(t1),max(t2),max(t3),max(t4),max(t5),max(t6),max(t7))

    for i in range(len(df)):
        y.append(1*x1[i]+3*x2_0[i]+1.5*x2_1[i]+0.25*x2_2[i]-0.01*x3[i]+0.2*t0[i]+0.15*t1[i]+0.25*t2[i]+0.44*t3[i]+0.13*t4[i]+0.03*t5[i]-0.14*t6[i]-0.04*t7[i]+ np.random.normal(0,0.5))

    out = pd.DataFrame({
        't0': t0,
        't1': t1,
        't2': t2,
        't3': t3,
        't4': t4,
        't5': t5,
        't6': t6,
        't7': t7,
        'y': y
    })

    df_out = pd.concat([df, out], axis=1)
    return df_out

alpha0 = 3
alpha1 = 3
train = pd.read_csv('data/profile_train_vec.csv')
A=np.eye(len(train), len(train))
for i in range(len(train)):
    for j in range(i+1, len(train)):
        a = np.array(train.iloc[i, 1:])
        b = np.array(train.iloc[j, 1:])
        logit = np.exp(alpha0-alpha1*np.linalg.norm([a,b]))
        friend = np.random.binomial(1, logit/(1+logit))
        A[i][j], A[j][i] = friend, friend

A = pd.DataFrame(A)
A.to_csv('data/usernet_train.csv',index=False)