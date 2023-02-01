import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np


X = pd.read_csv('task1.csv', header=None)  # читаем csv файл с данными, header - убрать заголовок
pca = PCA(n_components=2, svd_solver='full')  # вызываем расчётную нашу функцию, кол-во компонент
pca.fit(X)  # вычисляются средние значения и отклонения по каждому предиктору
scores = pca.transform(X)  # трансформирует все признаки в соответствии с соответствующими средними и отклонениями
scores_df = pd.DataFrame(scores, columns=['PC1', 'PC2'])  # создаём дата-фрейм
print(round(scores_df, 3))
print(pca.singular_values_)
print(round((pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]), 3))  # объяснённая дисперсия
# по 2-м компонентам
plt.scatter(scores_df.PC1, scores_df.PC2)
plt.show()  # на графике видно, что объекты по кучкам сгруппированы


# F - матрица векторов весов, Z - матрица векторов счётов
F = pd.read_csv('ves.csv', delimiter=';')  # просто считываем файл, указываем разделитель, по умолчанию он другой
Z = pd.read_csv('schet.csv', delimiter=';')
Fch = np.dot(Z,F.T)  # это просто умножение матрицы счётов на транспонированную матрицу весов как в теории
plt.matshow(Fch)  # выше мы получили точки х и у (как координаты) и по ним строим картинку, причём как бы
# в картинке есть зависимость между цветом и значением
plt.show()






