import pandas as pd
import numpy as np

'''

根据用户的相似性生成用户网络

'''

def generate_network(dt):
    A_dt = np.eye(len(dt), len(dt))
    for i in range(len(train)):
        for j in range(i + 1, len(train)):
            a = np.array(train.iloc[i, 1:])
            b = np.array(train.iloc[j, 1:])
            logit = np.exp(alpha0 - alpha1 * np.linalg.norm([a, b]))
            friend = np.random.binomial(1, logit / (1 + logit))
            A_dt[i][j], A_dt[j][i] = friend, friend
    return pd.DataFrame(A_dt)

alpha0 = 3
alpha1 = 3
train = pd.read_csv('data/Uprofile_train.csv')
test = pd.read_csv('data/Uprofile_test.csv')

A_train = generate_network(train)
A_train.to_csv('data/Unet_train.csv', index=False)
A_test = generate_network(test)
A_test.to_csv('data/Unet_test.csv', index=False)

y0_train = pd.DataFrame(np.random.randint(0,2,size=len(train)), columns=['y(t-1)']) # y(t-1)
y0_train.to_csv('data/y_train.csv',index=False)
y0_test = pd.DataFrame(np.random.randint(0,2,size=len(test)), columns=['y(t-1)']) # y(t-1)
y0_test.to_csv('data/y_test.csv',index=False)