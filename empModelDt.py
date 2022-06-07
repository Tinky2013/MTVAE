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
    emb = pd.read_csv(emb_path)
    emb_Ba = pd.read_csv(emb_ba_path)
    emb_fake = pd.read_csv(fake_emb)

    emb.columns = ['emb_'+str(i) for i in range(len(emb.columns))]
    emb_Ba.columns = ['emb_' + str(i) for i in range(len(emb_Ba.columns))]
    emb_fake.columns = ['emb_' + str(i) for i in range(len(emb_fake.columns))]

    dt1 = pd.concat([dt,emb], axis=1)
    dt2 = pd.concat([dt,emb_Ba], axis=1)
    dt3= pd.concat([dt,emb_fake], axis=1)

    vgae_emb4 = ' emb_0 + emb_1 + emb_2 + emb_3'
    vgaeBa_emb4 = ' emb_0 + emb_1 + emb_2 + emb_3'
    fake_emb4 = ' emb_0 + emb_1 + emb_2 + emb_3'

    # vgae_emb16 = ' emb_0 + emb_1 + emb_2 + emb_3 + emb_4 + emb_5 + emb_6 + emb_7 + emb_8 + emb_9 + emb_10 + emb_11 + emb_12 + emb_13 + emb_14 + emb_15'
    # vgaeBa_emb16 = ' emb_0 + emb_1 + emb_2 + emb_3 + emb_4 + emb_5 + emb_6 + emb_7 + emb_8 + emb_9 + emb_10 + emb_11 + emb_12 + emb_13 + emb_14 + emb_15'
    # fake_emb16 = ' emb_0 + emb_1 + emb_2 + emb_3 + emb_4 + emb_5 + emb_6 + emb_7 + emb_8 + emb_9 + emb_10 + emb_11 + emb_12 + emb_13 + emb_14 + emb_15'

    formula0 = 'y1 ~ y0 + influence_0'
    formula1 = 'y1 ~ y0 + influence_0 +' + vgae_emb4
    formula2 = 'y1 ~ y0 + influence_0 +' + vgaeBa_emb4
    formula3 = 'y1 ~ y0 + influence_0 +' + fake_emb4

    res = smf.ols(formula=formula0,data=dt).fit()
    inf0.append(res.params['influence_0'])
    ylag0.append(res.params['y0'])
    res = smf.ols(formula=formula1,data=dt1).fit()
    inf1.append(res.params['influence_0'])
    ylag1.append(res.params['y0'])
    res = smf.ols(formula=formula2,data=dt2).fit()
    inf2.append(res.params['influence_0'])
    ylag2.append(res.params['y0'])
    res = smf.ols(formula=formula3,data=dt3).fit()
    inf3.append(res.params['influence_0'])
    ylag3.append(res.params['y0'])


if __name__ == "__main__":
    data = 'B_0_3_0.3_100_N'
    Prob = 0.231
    # graph = 'test'
    inf0, ylag0 = [], []
    inf1, ylag1 = [], []
    inf2, ylag2 = [], []
    inf3, ylag3 = [], []
    for i in range(11,111):
        data_path = 'data/gendt/' + data + '/gendt_' + str(i) + '.csv'
        emb_path = 'save_emb/vgae/'+ data +'_zdim_4/emb_'+str(i)+'.csv'
        emb_ba_path = 'save_emb/vgae/'+ data +'Ba0.1_zdim_4/emb_'+str(i)+'.csv'
        fake_emb = 'save_emb/fake_dim4.csv'
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
    result.to_csv('result/'+data+'Ba0.1_'+str(Prob)+'p.csv',index=False)