import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency

def main():
    dt = pd.read_csv(data_path)

    emb1 = pd.read_csv(emb1_path)
    emb2 = pd.read_csv(emb2_path)
    emb1.columns = ['emb1_'+str(i) for i in range(len(emb1.columns))]
    emb2.columns = ['emb2_' + str(i) for i in range(len(emb2.columns))]
    dt1 = pd.concat([dt,emb1,emb2], axis=1)

    vgae_emb1_4 = ' emb1_0 + emb1_1 + emb1_2 + emb1_3'
    vgae_emb2_4 = ' emb2_0 + emb2_1 + emb2_2 + emb2_3'

    formula0 = 'y1 ~ y0 + influence_0'
    formula1 = 'y1 ~ y0 + influence_0 +' + vgae_emb1_4
    formula2 = 'y1 ~ y0 + influence_0 +' + vgae_emb2_4
    formula3 = 'y1 ~ y0 + influence_0 +' + vgae_emb1_4 + ' +' + vgae_emb2_4

    res = smf.ols(formula=formula0,data=dt).fit()
    inf0.append(res.params['influence_0'])
    ylag0.append(res.params['y0'])
    res = smf.ols(formula=formula1,data=dt1).fit()
    inf1.append(res.params['influence_0'])
    ylag1.append(res.params['y0'])
    res = smf.ols(formula=formula2,data=dt1).fit()
    inf2.append(res.params['influence_0'])
    ylag2.append(res.params['y0'])
    res = smf.ols(formula=formula3,data=dt1).fit()
    inf3.append(res.params['influence_0'])
    ylag3.append(res.params['y0'])


if __name__ == "__main__":
    data = 'J_0_3_0.5_100_N'
    Prob = 0.062
    # graph = 'test'
    inf0, ylag0 = [], []
    inf1, ylag1 = [], []
    inf2, ylag2 = [], []
    inf3, ylag3 = [], []
    for i in range(11,111):
        data_path = 'data/gendt/' + data + '/gendt_' + str(i) + '.csv'
        emb1_path = 'save_emb/vgae/'+ data +'_zdim_4/emb1_'+str(i)+'.csv'
        emb2_path = 'save_emb/vgae/' + data + '_zdim_4/emb2_' + str(i) + '.csv'
        # fake_emb = 'save_emb/fake.csv'
        main()
        print("regressing: ", i)
    result = pd.DataFrame({
        'inf0': inf0,
        'ylag0': ylag0,
        'inf1': inf1,
        'ylag1': ylag1,
        'inf2': inf2,
        'ylag2': ylag2,
        'inf3': inf3,
        'ylag3': ylag3,
    })
    result.to_csv('result/'+data+'_'+str(Prob)+'p.csv',index=False)