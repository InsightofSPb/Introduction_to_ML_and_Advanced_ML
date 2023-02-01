## Задание 1
Дана таблица \
![](https://github.com/InsightofSPb/Introduction_to_ML_and_Advanced_ML/blob/main/Tasks/regression.png?raw=true)
В таблице предиктор Х - это количество купленных товаров, отклик Y - время в магазине
Необходимо:
+ Найти параметры $\theta_1$, $\theta_2$, их оценки $SE(\theta_n)$
+ Найти границы доверительных интервалов

#### Подключим библиотеки, которые необходимы для этого и следующих заданий
```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
```
#### Вводим данные из таблицы и считаем параметры регрессии
```python
X = [13, 4, 11, 20]
Y1 = [8, 5, 6, 15]
Xc = np.mean(X)
Yc = np.mean(Y1)
print('Xc:',Xc,'Yc:', Yc)
A = 0
B = 0
for i in range(len(X)):
    A += (X[i] - Xc) * (Y1[i] - Yc)
    B += (X[i] - Xc) ** 2
teta1 = A / B
teta0 = Yc - teta1 * Xc
print('teta0:', round(teta0,2), 'teta1:',round(teta1, 2))
```
####
```python
t = 4.303  # квантиль распределения Стьюдента при t_0.975 и n - 2 (n = 4)
C = 0
D = 0
for i in range(len(X)):
    C += (Y1[i] - teta0 - teta1 * X[i]) ** 2
    D += (X[i] - Xc) ** 2
SEteta0 = np.sqrt(C / (len(X) -2)) * np.sqrt(1 / len(X) + Xc ** 2 / D)
SEteta1 = np.sqrt(C / (len(X) -2)) * np.sqrt(1 / D)
print('SEteta0:',round(SEteta0,2),'SEteta1:',round(SEteta1, 2))
teta1mi = teta1 - t * SEteta1  # находим границы доверительного интервала для каждого параметра
teta1ma = teta1 + t * SEteta1
teta0mi = teta0 - t * SEteta0
teta0ma = teta0 + t * SEteta0
RSE = np.sqrt( C / (len(X) - 2))  # расчёт оценки среднего квадратического отколонения ошибки
M = 0
for i in range(len(X)):
    M += (Y1[i] - Yc) ** 2
print('teta0mi:', teta0mi, 'teta0ma:', teta0ma, 'teta1mi:', teta1mi, 'teta1ma:', teta1ma)
print('R:', round(1 - C / M, 2), 'RSE:', RSE)  # вывод RSE и R - статистики
```
## Задание 2 
Дан файл с наблюдениями длительности нахождения человека в очереди в зависимости от количества людей в этой очереди (regr1.csv). Необходимо найти выборочное среднее предиктора, отклика, коэффициенты регрессии.
#### Считываем данные и рассчитываем средние значения
```python
D2 = pd.read_csv('regr1.csv')
X1 = np.array([D2['X']]).reshape(-1,1)
Y1 = D2['Y']
print(f'Xmin= {np.mean(X1)}, Ymin= {np.mean(Y1)}')
```
#### Обучаем модель и получаем параметры
```python
reg = LinearRegression()
reg.fit(X1,Y1)
teta1 = reg.coef_
teta0 = np.mean(Y1) - teta1 * np.mean(X1)
print(teta0, teta1)
```
## Задание 3
В файле candy.csv представлены данные, собранные путем голосования за самые лучшие (или, по крайней мере, самые популярные) конфеты Хэллоуина. Обучите модель линейной многомерной регрессии. В качестве предикторов выступают поля: chocolate, fruity, caramel, peanutyalmondy, nougat, crispedricewafer, hard, bar, pluribus, sugarpercent, pricepercent, отклик — winpercent.

В качестве тренировочного набора данных используйте данные из файла, за иключением следующих конфет: Boston Baked Beans, Peanut M&Ms. Обучите модель и выведите предсказание отклика winpercent для определённых конфет
#### Считываем файл и убираем лишние конфеты
```python
Data = pd.read_csv('candy-data.csv', index_col='competitorname')
Data = Data.drop(['Peanut M&Ms', 'Boston Baked Beans'])
X = pd.DataFrame(Data, columns=['chocolate','fruity','caramel','peanutyalmondy','nougat','crispedricewafer','hard','bar','pluribus','sugarpercent','pricepercent'])
Y = pd.DataFrame(Data, columns=['winpercent'])
```
#### Обучаем модель и выводим значения параметров регрессии
```python
X = X.values  # избавиться от предупреждения о том, что Х содержит не только значения, но и feature names
reg = LinearRegression()
reg.fit(X,Y)
print(reg.coef_, reg.score(X,Y))
```
#### Выводим значения откликов для конфеты по заданию
```python
Any = [1,1,1,1,1,1,0,0,0,0.261,0.273]
BBB = [0,0,0,1,0,0,0,0,1,0.31299999,0.51099998]
PM = [1,0,0,1,0,0,0,0,1,0.59299999,0.65100002]
M3 = [1,0,0,0,1,0,0,1,0,0.60399997,0.51099998]
A = reg.predict(np.array([M3]))
print(A)
```