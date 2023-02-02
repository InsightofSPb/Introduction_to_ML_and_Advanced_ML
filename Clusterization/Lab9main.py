import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

Data = pd.read_csv('objects.csv', delimiter=',', index_col='Object')
Data = Data.drop(columns='Cluster')
km = KMeans(n_clusters=3, init=np.array([[15.0, 8.5], [9.0, 12.83], [14.0, 5.0]]), max_iter=100, n_init=1)
dist = km.fit_transform(Data)
cl = km.labels_
D = np.mean(pd.DataFrame(dist, cl).loc[0])
print(cl)
print(D)
