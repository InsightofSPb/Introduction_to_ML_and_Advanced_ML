## Задание
В файле candy-data.csv представлены данные, собранные путем голосования за самые лучшие (или, по крайней мере, самые популярные) конфеты Хэллоуина. Обучите модель логистической регрессии. В качестве предикторов выступают поля: chocolate, fruity, caramel, peanutyalmondy, nougat, crispedricewafer, hard, bar, pluribus, sugarpercent, pricepercent, отклик — Y.

В качестве тренировочного набора данных используйте данные из файла, за иключением следующих конфет: Dots, Hersheys Milk Chocolate, Nik L Nip. Обучите модель.\
Обучите модель и выполните предсказание для всех конфет из файла candy-test.csv тестовых данных.

#### Импорт библиотек, обработка тренировочных данных, обучение 
```python
from numpy import round
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score

#  Training set
Data = pd.read_csv('candy-data.csv', index_col='competitorname')
Data = Data.drop(['Dots', 'Hersheys Milk Chocolate', 'Nik L Nip'])
Data = Data.drop(['winpercent'], axis=1)
Y = Data.loc[:, 'Y'].values
Data = Data.drop(['Y'], axis=1)
X = Data.values
reg = LogisticRegression(random_state = 2019, solver = 'lbfgs').fit(X, Y)  # параметры даны по заданию
```
#### Подготовка тренировочных данных, предсказание для двух нужных по заданию конфет
```python
#  Testing set
Testdate = pd.read_csv('candy-test.csv', index_col='competitorname')
TRSB = Testdate.loc['Tootsie Roll Snack Bars'].drop(['Y']).values.reshape(1, -1)
Wh = Testdate.loc['Warheads'].drop(['Y']).values.reshape(1, -1)
print(f'Tootsie Roll Snack Bars = {round(reg.predict_proba(TRSB)[:,1],3)}')
print(f'Tootsie Roll Snack Bars = {round(reg.predict_proba(Wh)[:,1],3)}')
```
#### Построение матрицы ошибок, расчёт Recall, Precision, AUC
```python
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

```