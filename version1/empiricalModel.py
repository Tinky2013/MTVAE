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
    emb.columns = ['emb_'+str(i) for i in range(len(emb.columns))]
    dt1 = pd.concat([dt,emb], axis=1)
    vgae_emb4 = ' emb_0 + emb_1 + emb_2 + emb_3'

    # leverage X
    embX = pd.read_csv(embX_path)
    embX.columns = ['emb_'+str(i) for i in range(len(embX.columns))]
    dt2 = pd.concat([dt, embX], axis=1)
    vgae_embX4 = ' emb_0 + emb_1 + emb_2 + emb_3'

    # Mtvae_Emb4 = ' mtvae_emb_0 + mtvae_emb_1 + mtvae_emb_2 + mtvae_emb_3'
    # Mtvae_Emb8 = Mtvae_Emb4 + ' + mtvae_emb_4 + mtvae_emb_5 + mtvae_emb_6 + mtvae_emb_7'
    # Mtvae_Emb16 = Mtvae_Emb8 + ' + mtvae_emb_8 + mtvae_emb_9 + mtvae_emb_10 + mtvae_emb_11 + mtvae_emb_12 + mtvae_emb_13 + mtvae_emb_14 + mtvae_emb_15'

    formula0 = 'y1 ~ y0 + influence_0'
    formula1 = 'y1 ~ y0 + influence_0 +' + vgae_emb4
    formula2 = 'y1 ~ y0 + influence_0 +' + vgae_embX4

    formula3 = 'y1 ~ y0 + influence_0 + x'
    formula4 = 'y1 ~ y0 + influence_0 + x +' + vgae_emb4
    formula5 = 'y1 ~ y0 + influence_0 + x +' + vgae_embX4

    res = smf.ols(formula=formula0,data=dt).fit()
    inf0.append(res.params['influence_0'])
    ylag0.append(res.params['y0'])
    res = smf.ols(formula=formula1,data=dt1).fit()
    inf1.append(res.params['influence_0'])
    ylag1.append(res.params['y0'])
    res = smf.ols(formula=formula2,data=dt2).fit()
    inf2.append(res.params['influence_0'])
    ylag2.append(res.params['y0'])

    res = smf.ols(formula=formula3,data=dt).fit()
    inf3.append(res.params['influence_0'])
    ylag3.append(res.params['y0'])
    res = smf.ols(formula=formula4,data=dt1).fit()
    inf4.append(res.params['influence_0'])
    ylag4.append(res.params['y0'])
    res = smf.ols(formula=formula5,data=dt2).fit()
    inf5.append(res.params['influence_0'])
    ylag5.append(res.params['y0'])

if __name__ == "__main__":
    graph = '0_1.5_1.5_3_X[0]_Z[0.4]'
    Prob = 0.492
    # graph = 'test'
    inf0, ylag0 = [], []
    inf1, ylag1 = [], []
    inf2, ylag2 = [], []
    inf3, ylag3 = [], []
    inf4, ylag4 = [], []
    inf5, ylag5 = [], []
    for i in range(11,21):
        data_path = 'data/gendt/'+graph+'/gendt_'+str(i)+'.csv'
        emb_path = 'save_emb/vgae_100/'+graph+'_zdim_4/emb_'+str(i)+'.csv'
        embX_path = 'save_emb/vgae_100/' + graph + '_zdim_4_withX/emb_' + str(i) + '.csv'
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
        'inf4': inf4,
        'ylag4': ylag4,
        'inf5': inf5,
        'ylag5': ylag5,
    })
    result.to_csv('result/'+graph+'_'+str(Prob)+'p.csv',index=False)