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

    reg_dt = pd.concat([dt, emb1, emb2], axis=1)

    f1_emb = ' emb1_0 + emb1_1 + emb1_2 + emb1_3'
    fn1_emb = ' emb1_0 + emb1_1 + emb1_2 + emb1_3 + emb1_4 + emb1_5 + emb1_6 + emb1_7'
    n1_emb = ' emb1_4 + emb1_5 + emb1_6 + emb1_7'
    f2_emb = ' emb2_0 + emb2_1 + emb2_2 + emb2_3'
    fn2_emb = ' emb2_0 + emb2_1 + emb2_2 + emb2_3 + emb2_4 + emb2_5 + emb2_6 + emb2_7'
    n2_emb = ' emb2_4 + emb2_5 + emb2_6 + emb2_7'

    # incorrect
    formula0 = 'y1 ~ y0 + influence_estim'
    # emb1 focal
    formula1 = 'y1 ~ y0 + influence_estim +' + f1_emb
    # emb1 focal+neighbor
    formula2 = 'y1 ~ y0 + influence_estim +' + fn1_emb
    # emb1 neighbor
    formula3 = 'y1 ~ y0 + influence_estim +' + n1_emb

    # emb2 focal
    formula4 = 'y1 ~ y0 + influence_estim +' + f2_emb
    # emb2 focal+neighbor
    formula5 = 'y1 ~ y0 + influence_estim +' + fn2_emb
    # emb2 neighbor
    formula6 = 'y1 ~ y0 + influence_estim +' + n2_emb

    # z+zn
    formula7 = 'y1 ~ y0 + influence_estim + z + zn'
    # z
    formula8 = 'y1 ~ y0 + influence_estim + z'
    # zn
    formula9 = 'y1 ~ y0 + influence_estim + zn'

    res = smf.ols(formula=formula0,data=reg_dt).fit()
    inf0.append(res.params['influence_estim'])
    res = smf.ols(formula=formula1,data=reg_dt).fit()
    inf1.append(res.params['influence_estim'])
    res = smf.ols(formula=formula2,data=reg_dt).fit()
    inf2.append(res.params['influence_estim'])
    res = smf.ols(formula=formula3,data=reg_dt).fit()
    inf3.append(res.params['influence_estim'])
    res = smf.ols(formula=formula4,data=reg_dt).fit()
    inf4.append(res.params['influence_estim'])
    res = smf.ols(formula=formula5,data=reg_dt).fit()
    inf5.append(res.params['influence_estim'])
    res = smf.ols(formula=formula6,data=reg_dt).fit()
    inf6.append(res.params['influence_estim'])
    res = smf.ols(formula=formula7,data=reg_dt).fit()
    inf7.append(res.params['influence_estim'])
    res = smf.ols(formula=formula8,data=reg_dt).fit()
    inf8.append(res.params['influence_estim'])
    res = smf.ols(formula=formula9,data=reg_dt).fit()
    inf9.append(res.params['influence_estim'])


if __name__ == "__main__":
    for st in ['0.18','0.33','0.5','0.69','0.92','1.2']:
        data = 'E_0_3_'+st+'_100_U'
        # graph = 'test'
        inf0, inf1, inf2, inf3, inf4, inf5, inf6, inf7, inf8, inf9 = [], [], [], [], [], [], [], [], [], []
        for i in range(11,111):
            data_path = 'data/gendt/' + data + '/gendt_' + str(i) + '.csv'
            emb1_path = 'save_emb/vgae/'+ data + 'A1Y1' + '_zdim_4/emb_'+str(i)+'.csv'
            emb2_path = 'save_emb/vgae/'+ data + 'A1Y0' + '_zdim_4/emb_'+str(i)+'.csv'
            main()
            print("regressing: ", i)
        result = pd.DataFrame({
            'inf0': inf0,
            'inf1': inf1,
            'inf2': inf2,
            'inf3': inf3,
            'inf4': inf4,
            'inf5': inf5,
            'inf6': inf6,
            'inf7': inf7,
            'inf8': inf8,
            'inf9': inf9,
        })
        result.to_csv('result/'+data+'.csv',index=False)