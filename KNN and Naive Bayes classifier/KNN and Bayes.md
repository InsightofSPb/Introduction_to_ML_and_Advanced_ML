## Задание 1
Реализовать расчёт расстояния от объекта до ближайшего соседа, используя евклидову метрику, Манхеттенское расстояние и растояние Чебышева
#### Реализована функция, рассчитывающая расстояния для двух объектов
```python
from numpy import *


def dist1(x1, x2):
    de = sqrt(sum([(i - j) ** 2 for i, j in zip(x1,x2)]))
    dm = sum([abs(i - j) for i, j in zip(x1,x2)])
    dch = max([i - j for i, j in zip(x1,x2)])
    print('Евклидово расстояние =',round(de,3))
    print('Манхеттенское расстояние =',round(dm,3))
    print('Расстояние Чебышёва =',round(dch,3))
```
## Задание 2
Доступна таблица некоторых синтетических данных, на основании которых необходимо выполнить классификацию нового объекта, с помощью метода <i>k</i> -ближайших соседей. Необходимо для объекта с координатами (52,18) вычислить расстояние до ближайшего соседа, используя евклидову метрику и метрику городских кварталов.
#### Функция расчёта расстояния между точками, можно возвращать необходимое расстояние
```python
def dist2(x1,x2):
    de = sqrt(sum([(i - j) ** 2 for i, j in zip(x1,x2)]))
    dm = sum([abs(i - j) for i, j in zip(x1,x2)])
    dch = max([i - j for i, j in zip(x1,x2)])
    return de
```
#### Вызов функции и рассчёт класса 0 или 1 при  k = 3. Исходные объекты принадлежат к разным классам
```python
T = array([52,18])
X = array([[31,19], [45,23], [15,46], [92,82], [78,29], [58,34],[25,19],[29,93],[84,82],[82,27]])
dev = []
dma = []
dch = []
for i in range(len(X)):
    m = dist2(T,X[i])
    dev.append(m)
print(dev)  # смотрим весь массив расстояний до каждого другого объекта
print('Класс "0":',round(dev[2] + dev[4],3),'Класс "1":', round(dev[0] + dev[1] + dev[3],4))
```

### Задание 2.1
Та же самая задача, но реализация с помощью sklearn
```python
import numpy as np
import pandas as pd
import sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier

Data = pd.read_csv('51.csv',index_col='id')
print(Data)

T = [52,18]
X = Data.iloc[:,:2].values
Y = Data.iloc[:,2]
print(X,Y)
neigh = KNeighborsClassifier(n_neighbors=1, p=1)
neigh.fit(X,Y)
neigh1 = sklearn.neighbors.NearestNeighbors(n_neighbors=1,p=1)
neigh1.fit(X,Y)
E1 = neigh1.kneighbors([T])
print('Класс предполагаемого объекта:', neigh.predict([T]))
print('Расстояние до ближайшего объекта по ЕвклМ:',round(float(E1[0]),3))
print('ID этого соседа:', int(E1[1] + 1))

neigh3 = KNeighborsClassifier(n_neighbors=3, p=1)
neigh3.fit(X,Y)
print('Класс предполагаемого объекта:', neigh3.predict([T]))
neigh31 = sklearn.neighbors.NearestNeighbors(n_neighbors=3,p=1)
neigh31.fit(X,Y)
E3 = neigh31.kneighbors([T])
print(E3)
print('ID этих соседей:', E3[1] + 1)
```


## Задание 3

#### Расчёт вероятности того, что письмо является спамом. Исходные данные ниже
![](https://github.com/InsightofSPb/Introduction_to_ML_and_Advanced_ML/blob/main/Tasks/Bayes0.png?raw=true)\
Найти вероятность того, что письмо является спамом, основываясь на приведенных выше данных и логарифм данной вероятности
```python
from numpy import *

Let = [11,26]
Letsp = Let[0]/ sum(Let)
letspln = log(Letsp)
print(round(Letsp,3),round(letspln,3))
```
#### Вычислить вероятность того, что если в письме есть слово win, слово dollar ( по отдельности), то такое письмо будет отнесено к спаму
```python
r = 1  # dollar, number of words not in dictionary
V = 8  # size of the dictionary
Spam = 69  # total number of spam words (words in spam letters)
Clear = 48  # total number of clear words
wins = 6  # win in spam
winc = 3  # win in clear
millions = 10
millionsc = 5
P_win_spam = round((r + wins) / (V + r + Spam),4)
P_dollars_spam = round((r) / (r + V + Spam), 4)
print(P_win_spam)
print(P_dollars_spam)
```
#### Классифицировать письмо "win million dollars"
```python
P_million_spam = round((r + 10) / (r + V + Spam),3)
F_spam = log(Let[0] / sum(Let)) + log((r + wins) / (r + V + Spam)) + log((r + millions) / (r + V + Spam)) + log((r) / (r + V + Spam))
F_clear = log(Let[1] / sum(Let)) + log((r + winc) / (r + V + Clear)) + log((r + millionsc) / (r + V + Clear)) + log((r) / (r + V + Clear))
P_spam_letter = 1 / (1 + exp(F_clear - F_spam))
print(round(F_spam, 3), round(F_clear, 3), round(P_spam_letter, 3))
```

## Задание 4
Даны две таблицы:
+ В первой содержатся  данные о классификации писем на «спам»/«не спам» и общее количество слов, входящих в эти группы;
+ Во второй представлены данные, по уникальным словам, и числу их вхождений в указанные группы.\
Задача состоит в том, чтобы построить модель наивого байесовского классификатора и определить класс, к которому будет отнесено письмо, которое содержит текст: "Purchase Bill Gift Unlimited Bonus Prize Cash". Кол-во слов приведено на картинке "Bayes"\
![](https://github.com/InsightofSPb/Introduction_to_ML_and_Advanced_ML/blob/main/Tasks/Bayes.png?raw=true)


#### Расчёт того, что письмо спам
```python
from numpy import log, exp

Let_spam = 19
Let_no_spam = 17
words_spam = 71
words_no_spam = 125
P_let_spam = Let_spam / (Let_no_spam + Let_spam)
print('Вероятность письма оказаться спамом:', round(P_let_spam,3))
```

#### Расчёт того, что письмо, содержащее слова "Purchase Bill Gift Unlimited Bonus Prize Cash" является спамом
```python
# Purchase Bill Gift Unlimited Bonus Prize Cash
r = 2
V = 10
words = [[9,0],[4,16],[4,1],[5,13],[1,5]]
F_let_spam = 0
F_let_no_spam = 0
for i in range(0,5):
    F_let_spam += log((1 + words[i][0]) / (r + V + words_spam))
    F_let_no_spam += log((1 + words[i][1]) / (r + V + words_no_spam))
F_let_spam += 2 * log((1 / (r + V + words_spam))) + log(P_let_spam)
F_let_no_spam += 2 * log((1 / (r + V + words_no_spam))) + log(1 - P_let_spam)
P_spam = 1 / (1 + exp(F_let_no_spam - F_let_spam))
print('Вероятность классифицировать письмо как спам:', round(P_spam,3))
print('F(спам):', round(F_let_spam,3), ';', 'F(не спам):', round(F_let_no_spam,3))


Pps = (1 + 9) / (r + V + 71)
Pbs = (1 + 4) / (r + V + 71)
Pgs = (1 + 4) / (r + V + 71)
Pbos = (1 + 5) / (r + V + 71)
Pprs = (1 + 1) / (r + V + 71)
Fpsp = log(P_let_spam) + log(Pps) + log(Pbs) + log(Pgs) + log(Pbos) + log(Pprs) + 2 * log(1 / (r + V + 71))
Ppns = (1 + 0) / (r + V + 125)
Pbns = (1 + 16) / (r + V + 125)
Pgns = (1 + 1) / (r + V + 125)
Pbons = (1 + 13) / (r + V + 125)
Pprns = (1 + 5) / (r + V + 125)
Fpnsp = log(1 - P_let_spam) + log(Ppns) + log(Pbns) + log(Pgns) + log(Pbons) + log(Pprns) + 2 * log(1 / (r + V + 125))
P_spam1 = 1 / (1 + exp(Fpnsp - Fpsp))
print(round(Fpsp,3), round(Fpnsp,3),round(P_spam1,3))
```

