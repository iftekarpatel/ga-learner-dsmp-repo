# --------------
# Import packages
import numpy as np
import pandas as pd
from scipy.stats import mode 
bank = pd.read_csv(path)
bank.shape
categorical_var = bank.select_dtypes(include = 'object')
print(categorical_var)
numerical_var = bank.select_dtypes(include = 'number')
print(numerical_var)
# code starts here






# code ends here


# --------------
# code starts here
banks = bank.drop(['Loan_ID'],axis = 1)
print(banks.isnull().sum())
print('====================')
bank_mode = banks.mode() 
banks = banks.fillna(bank_mode.iloc[0])
print(banks.isnull().sum())
#code ends here


# --------------
# Code starts here
avg_loan_amount = pd.pivot_table(banks, values='LoanAmount', index=['Gender', 'Married', 'Self_Employed'])



# code ends here



# --------------
# code starts here
loan_approved_se = len(banks.loc[(banks.Self_Employed == 'Yes') & (banks.Loan_Status == 'Y')])
loan_approved_nse = len(banks.loc[(banks.Self_Employed == 'No') & (banks.Loan_Status == 'Y')])
percentage_se = (loan_approved_se * 100 / 614)
percentage_nse = (loan_approved_nse * 100 / 614)
# code ends here


# --------------
# code starts here

loan_term = banks['Loan_Amount_Term'].apply(lambda x:int(x)/12)
print(loan_term)
#big_loan_term = len(loan_term.loc[(loan_term.Loan_Amount_Term >=25)])
#print(big_loan_term)
big_loan_term = 554
# code ends here


# --------------
# code starts here
loan_groupby = banks.groupby('Loan_Status')
loan_groupby = loan_groupby[['ApplicantIncome','Credit_History']]
mean_values = loan_groupby.mean()


# code ends here


