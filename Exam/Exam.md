# Формат экзамена
Дистацнионный, с системой прокторинга. 1,5 часа, хотя само задание спокойно делается за 40 минут. Помимо приведённого ниже кода было 4 вопроса по регрессии и логической регрессии.
## Задание на программирование
В данном упражнении вам предстоит решить задачу многокласовой классификации, используя в качестве тренировочного набора данных -- набор данных MNIST, содержащий образы рукописных цифр.
### 1. Используйте метод главных компонент для набора данных MNIST (train dataset объема 60000). Определите, какое минимальное количество главных компонент <i>M</i> необходимо использовать, чтобы доля объясненной дисперсии превышала 0.83 (была строго больше указанного значения).
```python
# Импортируем все необходимые библиотеки и сам датасет
from keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd

# Датасет и МГК
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape((X_train.shape[0], -1))
print(X_train.shape)

pca = PCA(n_components=56, svd_solver='full')
pca.fit(X_train)
print(f'Доля объяснённой дисперсии = {sum(pca.explained_variance_ratio_)}')
```

### 2. Найдите счеты, отвечающие найденным в предыдущем пункте <i>M</i> главным компонентам: PCA(n_components=M, svd_solver='full').
```python
print(f'Веса = {pca.singular_values_}')
```
### 3. Разделите полученную выборку (после сокращения размерности) случайным образом в отношении 70/30: train_test_split(X_train, y_train, test_size=0.3, random_state=126).
```python
X_pca = pca.transform(X_train)
X_train, X_test, Y_train, Y_test = train_test_split(X_pca, y_train, test_size=0.3, random_state=95)
```
### 4. Используя метод многоклассовой классификации One-vs-All OneVsRestClassifier(), обучите алгоритм случайного леса RandomForestClassifier() с параметрами criterion='gini', min_samples_leaf=10, max_depth=20, n_estimators=10, random_state=126. Выполните оценку с помощью тестовых данных. Обучите метод LogisticRegression, DecisionTreeClassifier.
```python
clf1 = OneVsRestClassifier(RandomForestClassifier(criterion='gini',min_samples_leaf=10,max_depth=20, n_estimators=10,random_state=95)).fit(X_train,Y_train)
pred1 = clf1.predict(X_test)
cm1 = confusion_matrix(Y_test,pred1)
print(cm1)
clf2 = OneVsRestClassifier(LogisticRegression(solver='lbfgs', random_state=95)).fit(X_train,Y_train)
pred2 = clf2.predict(X_test)
cm2 = confusion_matrix(Y_test,pred2)
print(cm2)
clf3 = OneVsRestClassifier(DecisionTreeClassifier(criterion='gini',min_samples_leaf=10, max_depth=20,random_state=95)).fit(X_train,Y_train)
pred3 = clf3.predict(X_test)
cm3 = confusion_matrix(Y_test,pred3)
print(cm3)
```
### 5. Примените полученное ранее преобразование метода главных компонент к новым данным (идентификаторы строк соответствуют файлам изображений).
```python
from google.colab import files
uploaded = files.upload()

import io
df2 = pd.read_csv(io.BytesIO(uploaded['pred_for_task.csv']), index_col='FileName')
print(df2.head())

df2ans = df2.loc[:,'Label'].values
df2tr = df2.drop(['Label'],axis=1).values

trans =  pca.transform(df2tr)
print(trans)
```
### 6. Выполните предсказание для указанных изображений цифр с помощью обученных алгоритмов.
```python
f3 = trans[2, :].reshape(1, -1)
f1 = trans[0, :].reshape(1, -1)
f23 = trans[22, :].reshape(1, -1)
P1 = max(max(clf1.predict_proba(f3)))
P2 = max(max(clf2.predict_proba(f1)))
P3 = max(max(clf3.predict_proba(f23)))
print(f'Вероятность отнесения файла 3 к назначенному классу = {round(P1,3)}')
print(f'Вероятность отнесения файла 1 к назначенному классу = {round(P2,3)}')
print(f'Вероятность отнесения файла 23 к назначенному классу = {round(P3,3)}')
```