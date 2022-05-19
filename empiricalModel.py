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
# avoid multicollinearity
dv = dv.drop('cluster_2')
dv = dv.drop('t1')
dv = dv.drop('t4')
dv = dv.drop('t7')

dv = dv.drop('y0')
dv = dv.drop('y1')
formula = 'y1 ~ y0'
for i in dv:
    formula = formula + ' + ' + i

print(formula)

Full_Emb = ' vgae_emb_0 + vgae_emb_1 + vgae_emb_2 + vgae_emb_3 + mtvae_emb_0 + mtvae_emb_1 + mtvae_emb_2 + mtvae_emb_3'
Mtvae_Emb = ' mtvae_emb_0 + mtvae_emb_1 + mtvae_emb_2 + mtvae_emb_3'
Vgae_Emb = ' vgae_emb_0 + vgae_emb_1 + vgae_emb_2 + vgae_emb_3'

# formula0 = 'y1 ~ y0 + gender + cluster_0 + cluster_1 + age + t0 + t2 + t3 + t5 + t6 + ' + Full_Emb
# formula1 = 'y1 ~ y0 + gender + cluster_0 + cluster_1 + age + t0 + t2 + t3 + t5 + t6 + ' + Mtvae_Emb
# formula2 = 'y1 ~ y0 + gender + cluster_0 + cluster_1 + age + t0 + t2 + t3 + t5 + t6 + ' + Vgae_Emb
# formula3 = 'y1 ~ y0 + gender + cluster_0 + cluster_1 + age + t0 + t2 + t3 + t5 + t6'

formula4 = 'y1 ~ y0 + gender + cluster_0 + cluster_1 + age + t0 + ' + Full_Emb
formula5 = 'y1 ~ y0 + gender + cluster_0 + cluster_1 + age + t0 + ' + Mtvae_Emb
formula6 = 'y1 ~ y0 + gender + cluster_0 + cluster_1 + age + t0 + ' + Vgae_Emb
formula7 = 'y1 ~ y0 + gender + cluster_0 + cluster_1 + age + t0'


res = smf.logit(formula=formula4,data=dt).fit()
print(res.summary())
