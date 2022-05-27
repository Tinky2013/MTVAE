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
    emb = pd.DataFrame(np.repeat(emb.values, K, axis=0))
    emb.columns = ['emb_'+str(i) for i in range(len(emb.columns))]
    dt1 = pd.concat([dt,emb], axis=1)

    pfm_emb4 = ' emb_0 + emb_1 + emb_2 + emb_3'

    formula0 = 'y1 ~ y0 + influence'
    formula1 = 'y1 ~ y0 + influence +' + pfm_emb4

    res = smf.ols(formula=formula0,data=dt).fit()
    inf0.append(res.params['influence'])
    ylag0.append(res.params['y0'])
    #print(res.summary())
    res = smf.ols(formula=formula1,data=dt1).fit()
    inf1.append(res.params['influence'])
    ylag1.append(res.params['y0'])
    #print(res.summary())


if __name__ == "__main__":
    K=200
    data = 'A_100_'+str(K)
    # graph = 'test'
    inf0, ylag0 = [], []
    inf1, ylag1 = [], []

    for i in range(11,111):
        data_path = 'data/genRec/' + data + '/genRec_' + str(i) + '.csv'
        emb_path = 'save_emb/pfm/'+ data +'_zdim_4/emb_'+str(i)+'.csv'

        main()
        print("regressing: ", i)
    result = pd.DataFrame({
        'inf0': inf0,
        'ylag0': ylag0,
        'inf1': inf1,
        'ylag1': ylag1,
    })
    result.to_csv('result/'+data+'.csv',index=False)