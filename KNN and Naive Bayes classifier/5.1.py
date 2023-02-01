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
