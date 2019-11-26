# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# code starts here
df = pd.read_csv(path)
p_a = df[df['fico']>700].shape[0]/df.shape[0]
p_b = df[df['purpose'] == 'debt_consolidation'].shape[0]/df.shape[0]
df1 = df[df['purpose'] == 'debt_consolidation']
p_a_b=(p_b and p_a)/p_a
result = p_a_b == p_b
# code ends here


# --------------
# code starts here
prob_lp = df[df['paid.back.loan'] =='Yes'].shape[0]/df.shape[0]
prob_cs = df[df['credit.policy'] =='Yes'].shape[0]/df.shape[0]
new_df = df[df['paid.back.loan'] =='Yes']
prob_pd_cs = new_df[new_df['credit.policy'] =='Yes'].shape[0]/new_df.shape[0]
bayes = 0.8684824902723736
# code ends here


# --------------
# code starts here
df1 = df[df['paid.back.loan'] =='No']


# code ends here


# --------------
# code starts here
inst_mean = df['installment'].mean()
inst_median = df['installment'].median()


# code ends here


