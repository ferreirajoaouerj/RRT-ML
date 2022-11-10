import pandas as pd


df_train = pd.read_csv('_train.csv')
df_val = pd.read_csv('_train.csv')

del df_train['i']
del df_val['i']

df_train = df_train.rename(columns={
    'sin_theta_i': 'cos_theta_i',
    'cos_theta_i': 'sin_theta_i',
    'sin_theta_f': 'cos_theta_f',
    'cos_theta_f': 'sin_theta_f',
    'sin_theta': 'cos_theta',
    'cos_theta': 'sin_theta',
})

df_val = df_val.rename(columns={
    'sin_theta_i': 'cos_theta_i',
    'cos_theta_i': 'sin_theta_i',
    'sin_theta_f': 'cos_theta_f',
    'cos_theta_f': 'sin_theta_f',
    'sin_theta': 'cos_theta',
    'cos_theta': 'sin_theta',
})

df_train.to_csv('train.csv')
df_val.to_csv('val.csv')
