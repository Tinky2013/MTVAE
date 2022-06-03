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

    formula1 = 'influence_0 ~' + vgae_emb4
    res = smf.ols(formula=formula1, data=dt1).fit()
    print(res.summary())


if __name__ == "__main__":
    data = 'B_0_3_0.3_100_N'
    for i in range(11,111):
        data_path = 'data/gendt/' + data + '/gendt_' + str(i) + '.csv'
        emb_path = 'save_emb/vgae/'+ data +'_zdim_4/emb_'+str(i)+'.csv'
        main()
