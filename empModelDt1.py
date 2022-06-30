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
    #emb = pd.read_csv(emb_path)
    emb_Ba = pd.read_csv(emb_ba_path)
    emb_Ba1 = pd.read_csv(emb_ba1_path)
    emb_Ba2 = pd.read_csv(emb_ba2_path)
    emb_fake = pd.read_csv(fake_emb)

    #emb.columns = ['emb_'+str(i) for i in range(len(emb.columns))]
    emb_Ba.columns = ['embba_' + str(i) for i in range(len(emb_Ba.columns))]
    emb_Ba1.columns = ['embba1_' + str(i) for i in range(len(emb_Ba1.columns))]
    emb_Ba2.columns = ['embba2_' + str(i) for i in range(len(emb_Ba2.columns))]
    emb_fake.columns = ['embf_' + str(i) for i in range(len(emb_fake.columns))]

    #reg_dt = pd.concat([dt, emb, emb_Ba, emb_Ba1, emb_Ba2, emb_fake], axis=1)
    reg_dt = pd.concat([dt, emb_Ba, emb_Ba1, emb_Ba2, emb_fake], axis=1)

    fake_emb4 = ' embf_0 + embf_1 + embf_2 + embf_3'
    #vgae_emb4 = ' emb_0 + emb_1 + emb_2 + emb_3'
    Ba_emb4 = ' embba_0 + embba_1 + embba_2 + embba_3'
    Ba1_emb4 = ' embba1_0 + embba1_1 + embba1_2 + embba1_3'
    Ba2_emb4 = ' embba2_0 + embba2_1 + embba2_2 + embba2_3'
    Bacon_emb4 = ' embba_0 + embba_1 + embba_2 + embba_3 + embba1_0 + embba1_1 + embba1_2 + embba1_3'


    # vgae_emb16 = ' emb_0 + emb_1 + emb_2 + emb_3 + emb_4 + emb_5 + emb_6 + emb_7 + emb_8 + emb_9 + emb_10 + emb_11 + emb_12 + emb_13 + emb_14 + emb_15'
    # vgaeBa_emb16 = ' emb_0 + emb_1 + emb_2 + emb_3 + emb_4 + emb_5 + emb_6 + emb_7 + emb_8 + emb_9 + emb_10 + emb_11 + emb_12 + emb_13 + emb_14 + emb_15'
    # fake_emb16 = ' emb_0 + emb_1 + emb_2 + emb_3 + emb_4 + emb_5 + emb_6 + emb_7 + emb_8 + emb_9 + emb_10 + emb_11 + emb_12 + emb_13 + emb_14 + emb_15'

    formula0 = 'y1 ~ y0 + influence_0'
    formula1 = 'y1 ~ y0 + influence_0 +' + fake_emb4
    #formula2 = 'y1 ~ y0 + influence_0 +' + vgae_emb4
    formula3 = 'y1 ~ y0 + influence_0 +' + Ba_emb4
    formula4 = 'y1 ~ y0 + influence_0 +' + Ba1_emb4
    formula5 = 'y1 ~ y0 + influence_0 +' + Bacon_emb4
    formula6 = 'y1 ~ y0 + influence_0 +' + Ba2_emb4
    formula7 = 'y1 ~ y0 + influence_0 + z'

    res = smf.ols(formula=formula0,data=reg_dt).fit()
    inf0.append(res.params['influence_0'])
    yt0.append(res.params['y0'])
    res = smf.ols(formula=formula1,data=reg_dt).fit()
    inf1.append(res.params['influence_0'])
    yt1.append(res.params['y0'])
    #res = smf.ols(formula=formula2,data=reg_dt).fit()
    #inf2.append(res.params['influence_0'])
    res = smf.ols(formula=formula3,data=reg_dt).fit()
    inf3.append(res.params['influence_0'])
    yt3.append(res.params['y0'])
    res = smf.ols(formula=formula4,data=reg_dt).fit()
    inf4.append(res.params['influence_0'])
    yt4.append(res.params['y0'])
    res= smf.ols(formula=formula5,data=reg_dt).fit()
    inf5.append(res.params['influence_0'])
    yt5.append(res.params['y0'])
    res = smf.ols(formula=formula6,data=reg_dt).fit()
    inf6.append(res.params['influence_0'])
    yt6.append(res.params['y0'])
    res = smf.ols(formula=formula7,data=reg_dt).fit()
    inf7.append(res.params['influence_0'])
    yt7.append(res.params['y0'])

if __name__ == "__main__":
    data = 'B_0_3_0.3_100_N'
    Prob = 0.231
    # graph = 'test'
    #inf0, inf1, inf2, inf3, inf4, inf5, inf6, inf7 = [], [], [], [], [], [], [], []
    inf0, inf1, inf3, inf4, inf5, inf6, inf7 = [], [], [], [], [], [], []
    yt0, yt1, yt3, yt4, yt5, yt6, yt7 = [], [], [], [], [], [], []
    for i in range(11,111):
        data_path = 'data/gendt/' + data + '/gendt_' + str(i) + '.csv'
        #emb_path = 'save_emb/vgae/'+ data +'_zdim_4/emb_'+str(i)+'.csv'
        emb_ba_path = 'save_emb/vgae/'+ data + 'T1A1Y0' + '_zdim_4/emb_'+str(i)+'.csv'
        emb_ba1_path = 'save_emb/vgae/' + data + 'T1A0Y0' + '_zdim_4/emb_' + str(i) + '.csv'
        emb_ba2_path = 'save_emb/vgae/' + data + 'T1A0Y1' + '_zdim_4/emb_' + str(i) + '.csv'
        fake_emb = 'save_emb/fake_dim4.csv'
        main()
        print("regressing: ", i)
    result = pd.DataFrame({
        'inf0': inf0,
        'inf1': inf1,
        #'inf2': inf2,
        'inf3': inf3,
        'inf4': inf4,
        'inf5': inf5,
        'inf6': inf6,
        'inf7': inf7,
        'yt0': yt0,
        'yt1': yt1,
        'yt3': yt3,
        'yt4': yt4,
        'yt5': yt5,
        'yt6': yt6,
        'yt7': yt7,
    })
    result.to_csv('result/'+data+'con'+'_'+str(Prob)+'p.csv',index=False)