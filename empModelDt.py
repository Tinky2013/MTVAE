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
    emb_fake = pd.read_csv(fake_emb)
    emb.columns = ['emb_'+str(i) for i in range(len(emb.columns))]
    emb_fake.columns = ['emb_' + str(i) for i in range(len(emb_fake.columns))]
    dt1 = pd.concat([dt,emb], axis=1)
    dt2 = pd.concat([dt,emb_fake], axis=1)
    vgae_emb4 = ' emb_0 + emb_1 + emb_2 + emb_3'
    fake_emb4 = ' emb_0 + emb_1 + emb_2 + emb_3'

    # leverage X
    # embX = pd.read_csv(embX_path)
    # embX.columns = ['emb_'+str(i) for i in range(len(embX.columns))]
    # dt2 = pd.concat([dt, embX], axis=1)
    # vgae_embX4 = ' emb_0 + emb_1 + emb_2 + emb_3'

    # Mtvae_Emb4 = ' mtvae_emb_0 + mtvae_emb_1 + mtvae_emb_2 + mtvae_emb_3'
    # Mtvae_Emb8 = Mtvae_Emb4 + ' + mtvae_emb_4 + mtvae_emb_5 + mtvae_emb_6 + mtvae_emb_7'
    # Mtvae_Emb16 = Mtvae_Emb8 + ' + mtvae_emb_8 + mtvae_emb_9 + mtvae_emb_10 + mtvae_emb_11 + mtvae_emb_12 + mtvae_emb_13 + mtvae_emb_14 + mtvae_emb_15'

    formula0 = 'y1 ~ y0 + influence_0'
    formula1 = 'y1 ~ y0 + influence_0 +' + vgae_emb4
    formula2 = 'y1 ~ y0 + influence_0 +' + fake_emb4

    res = smf.ols(formula=formula0,data=dt).fit()
    inf0.append(res.params['influence_0'])
    ylag0.append(res.params['y0'])
    res = smf.ols(formula=formula1,data=dt1).fit()
    inf1.append(res.params['influence_0'])
    ylag1.append(res.params['y0'])
    res = smf.ols(formula=formula2,data=dt2).fit()
    inf2.append(res.params['influence_0'])
    ylag2.append(res.params['y0'])


if __name__ == "__main__":
    data = 'A_0_0.5_0.1_100_N'
    Prob = 0.094
    # graph = 'test'
    inf0, ylag0 = [], []
    inf1, ylag1 = [], []
    inf2, ylag2 = [], []
    for i in range(11,111):
        data_path = 'data/gendt/' + data + '/gendt_' + str(i) + '.csv'
        emb_path = 'save_emb/vgae/'+ data +'_zdim_4/emb_'+str(i)+'.csv'
        fake_emb = 'save_emb/fake.csv'
        main()
        print("regressing: ", i)
    result = pd.DataFrame({
        'inf0': inf0,
        'ylag0': ylag0,
        'inf1': inf1,
        'ylag1': ylag1,
        'inf2': inf2,
        'ylag2': ylag2,
    })
    result.to_csv('result/'+data+'_'+str(Prob)+'p.csv',index=False)