import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression, LogisticRegression

dt = pd.read_csv('data/gendt_train.csv')
emb_vgae = pd.read_csv('save_emb/vgae_dim_4.csv')
emb_mtvae = pd.read_csv('save_emb/mtvae_dim_4.csv')



emb_vgae.columns = ['vgae_emb_'+str(i) for i in range(len(emb_vgae.columns))]
emb_mtvae.columns = ['mtvae_emb_'+str(i) for i in range(len(emb_mtvae.columns))]
dt = pd.concat([dt,emb_vgae,emb_mtvae], axis=1)

dt.drop(labels=['Id'], axis=1, inplace=True)
#dt.drop(labels=['cluster_2','t1','t4','t7'], axis=1, inplace=True)
dt.drop(labels=['cluster_2'], axis=1, inplace=True)
y = dt['y1']
X = dt.drop(labels=['y1'], axis=1)


clf = LogisticRegression(random_state=100).fit(X, y)

for i in range(len(X.columns)):
    print(X.columns[i], clf.coef_[0][i])
