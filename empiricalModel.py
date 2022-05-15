import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency

dt = pd.read_csv('data/gendt_train.csv')
emb_vgae = pd.read_csv('save_emb/vgae_dim_4.csv')
emb_mtvae = pd.read_csv('save_emb/mtvae_dim_4.csv')

emb_vgae.columns = ['vgae_emb_'+str(i) for i in range(len(emb_vgae.columns))]
emb_mtvae.columns = ['mtvae_emb_'+str(i) for i in range(len(emb_mtvae.columns))]
dt = pd.concat([dt,emb_vgae,emb_mtvae], axis=1)

dv = dt.columns
dv = dv.drop('Id')
dv = dv.drop('cluster_2')
dv = dv.drop('y0')
dv = dv.drop('y1')
formula = 'y1 ~ y0'
for i in dv:
    formula = formula + ' + ' + i

res = smf.logit(formula='y0 ~ age + cluster_0 + cluster_1', data=dt).fit()
print(res.summary())
