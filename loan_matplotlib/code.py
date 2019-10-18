# --------------
#Importing header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv(path)
#Code starts here
loan_status = data['Loan_Status'].value_counts(ascending=False)
loan_status.plot(kind="bar")


# --------------
#Code starts here



 
property_and_loan= data.groupby(['Property_Area','Loan_Status']).size().unstack()
property_and_loan.plot(kind='bar',stacked=False, figsize=(15,10))
# Label X-axes and Y-axes
plt.xlabel('Property Area')
plt.ylabel('Loan Status')
plt.xticks(rotation=45)
plt.show()


# --------------
#Code starts here




education_and_loan = data.groupby(['Education','Loan_Status']).size().unstack()
education_and_loan.plot(kind='bar',stacked=True, figsize=(15,10))
# Label X-axes and Y-axes
plt.xlabel('Education Status')
plt.ylabel('Loan Status')
plt.xticks(rotation=45)
plt.show()


# --------------
#Code starts here
graduate = data[data['Education'] == 'Graduate']
not_graduate = data[data['Education'] == 'Not Graduate']


graduate.plot.density(x='Education', y='LoanAmount',label = 'Graduate')
not_graduate.plot.density(x='Education', y='LoanAmount', label = 'Not Graduate')


#Code ends here

#For automatic legend display
plt.legend()


# --------------
#Code starts here
fig, (ax_1, ax_2, ax_3) = plt.subplots(nrows = 3 , ncols = 1, figsize=(20,10))
ax_1.scatter('ApplicantIncome','LoanAmount')
ax_1.set_title('ApplicantIncome')
ax_2.scatter('CoapplicantIncome','LoanAmount')
ax_2.set_title('CoapplicantIncome')
data['TotalIncome'] = data['ApplicantIncome'] + data['CoapplicantIncome'] 
ax_3.scatter('TotalIncome','LoanAmount')
ax_3.set_title('TotalIncome')


