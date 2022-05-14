import pandas as pd
import numpy as np

'''

T根据X生成，Y根据T和X生成
X包括gender(X1), cluster(X2, dummies), age(X3)

T0=5*X1+5*(X2=1)+0.1*X3
T1=5*X1+5*(X2=2)+0.1*X3
T2=5*X1+5*(X2=3)+0.1*X3
T3=-5*X1+5*(X2=1)+0.1*X3
T4=-5*X1+5*(X2=2)+0.1*X3
T5=-5*X1+5*(X2=3)+0.1*X3
T6=5*X1+0.1*X3
T7=-5*X1+0.005*X3*X3
T0~T7按列normalize到[0,1]

Y=1*X1+3*(X2=1)+1.5*(X2=2)+0.25*(X2=3)-0.01*X3
    +alpha*T

'''

def mx(vec):
    Min = min(vec)
    Max = max(vec)
    return (vec-min(vec))/(max(vec)-min(vec)+1e-3) + 1e-6

def generate(df):
    x1 = df['gender']
    x2_0 = df['cluster_0']
    x2_1 = df['cluster_1']
    x2_2 = df['cluster_2']
    x3 = df['age']

    t0, t1, t2, t3, t4, t5, t6, t7, y = [], [], [], [], [], [], [], [], []
    for i in range(len(df)):
        t0.append(5 * x1[i] + 5 * x2_0[i] + 0.1 * x3[i] + np.random.normal(0,1))
        t1.append(5 * x1[i] + 5 * x2_1[i] + 0.1 * x3[i] + np.random.normal(0,1))
        t2.append(5 * x1[i] + 5 * x2_2[i] + 0.1 * x3[i] + np.random.normal(0,1))
        t3.append(-5 * x1[i] + 5 * x2_0[i] + 0.1 * x3[i] + np.random.normal(0,1))
        t4.append(-5 * x1[i] + 5 * x2_1[i] + 0.1 * x3[i] + np.random.normal(0,1))
        t5.append(-5 * x1[i] + 5 * x2_2[i] + 0.1 * x3[i] + np.random.normal(0,1))
        t6.append(5 * x1[i] + 0.1 * x3[i] + np.random.normal(0,1))
        t7.append(-5 * x1[i] + 0.005 * x3[i] * x3[i] + np.random.normal(0,1))

    # normalize and avoid exact 0 or 1
    t0, t1, t2, t3, t4, t5, t6, t7 = mx(t0), mx(t1), mx(t2), mx(t3), mx(t4), mx(t5), mx(t6), mx(t7)
    print(min(t0), min(t1), min(t2), min(t3), min(t4), min(t5), min(t6), min(t7))
    print(max(t0),max(t1),max(t2),max(t3),max(t4),max(t5),max(t6),max(t7))

    for i in range(len(df)):
        y.append(1*x1[i]+3*x2_0[i]+1.5*x2_1[i]+0.25*x2_2[i]-0.01*x3[i]+0.2*t0[i]+0.15*t1[i]+0.25*t2[i]+0.44*t3[i]+0.13*t4[i]+0.03*t5[i]-0.14*t6[i]-0.04*t7[i]+ np.random.normal(0,0.5))

    out = pd.DataFrame({
        't0': t0,
        't1': t1,
        't2': t2,
        't3': t3,
        't4': t4,
        't5': t5,
        't6': t6,
        't7': t7,
        'y': y
    })

    df_out = pd.concat([df, out], axis=1)
    return df_out

train = pd.read_csv('data/Uprofile_train.csv')
generate(train).to_csv('data/gen_train.csv',index=False)