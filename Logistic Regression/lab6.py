from numpy import round
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score

#  Training set
Data = pd.read_csv('candy-data.csv', index_col='competitorname')
Data = Data.drop(['Dots', 'Hersheys Milk Chocolate', 'Nik L Nip'])
Data = Data.drop(['winpercent'], axis = 1)
Y = Data.loc[:, 'Y'].values
Data = Data.drop(['Y'], axis = 1)
X = Data.values
reg = LogisticRegression(random_state = 2019, solver = 'lbfgs').fit(X, Y)

#  Testing set
Testdate = pd.read_csv('candy-test.csv', index_col='competitorname')
TRSB = Testdate.loc['Tootsie Roll Snack Bars'].drop(['Y']).values.reshape(1, -1)
Wh = Testdate.loc['Warheads'].drop(['Y']).values.reshape(1, -1)
print(f'Tootsie Roll Snack Bars = {round(reg.predict_proba(TRSB)[:,1],3)}')
print(f'Tootsie Roll Snack Bars = {round(reg.predict_proba(Wh)[:,1],3)}')

#  error matrix
matrix = confusion_matrix(reg.predict(Testdate.loc[:, : 'pricepercent'].values),
                          Testdate.loc[:, 'Y'].values.reshape(-1, 1))
TPR = round(matrix[1,1]/ (matrix[1,1] + matrix[0,1]), 3)
Pr = round(matrix[1,1] / (matrix[1,1] + matrix[1,0]), 3)
print('Matrix')
print(matrix)
print(f' TPR = {TPR}')
print(f' Precision = {Pr}')

AUC = round(roc_auc_score((Testdate.loc[:, 'Y'].values.reshape(-1, 1)), reg.decision_function(Testdate.loc[:, : 'pricepercent'].values)),3)
print(f'AUC = {AUC}')




