from numpy import *

X1 = [1 , 2]
X2 = [0, 3]
X3 = [-1, 1]

d1 = sqrt(sum([(i - j) ** 2 for i,j in zip(X1, X2)]))
d2 = sqrt(sum([(i - j) ** 2 for i,j in zip(X2, X3)]))
d3 = sqrt(sum([(i - j) ** 2 for i,j in zip(X1, X3)]))
print(sum([d1,d2,d3]))
print(f'Внутрикластерное расстоение = {round(2 * sum([d1, d2, d3]),3)}, Среднее расстояние = {round(2 * sum([d1, d2, d3]) / 3, 3)}')

C11 = sum([X1[0], X2[0], X3[0]]) / 3
C12 = sum([X1[1], X2[1], X3[1]]) / 3
print(f'Первая координата = {C11}, вторая координата = {C12}')
C1 = [C11, C12]
C2 = [2, -3]
X4 = [4, 0]

d1 = sum([(i - j) ** 2 for i,j in zip(X4, C1)])
d2 = sum([(i - j) ** 2 for i,j in zip(X4, C2)])
if min(d1, d2) == d1:
    print(f'Кластер = C1, расстояние = {d1} ')
else:
    print(f'Кластер = C2, расстояние = {d2} ')


K1 = [[1, 2], [0, 3], [-1, 1]]
K2 = [[0, -4], [4, -2]]
K11, K12, K21, K22 = zeros(4)

for i in range(len(K1)):
    K11 += K1[i][0]
    K12 += K1[i][1]
for i in range(len(K2)):
    K21 += K2[i][0]
    K22 += K2[i][1]
print(f'Центроид С1 имеет координаты = {K11 / 3 , K12 / 3}')
print(f'Центроид С2 имеет координаты = {K21 / 2 , K22 / 2}')
C1 = [K11 / 3, K12 / 3]
C2 = [K21 / 2, K22 / 2]
print(f'Центроидный метод: {round(sqrt(sum([(i - j) ** 2 for i,j in zip(C1, C2)])), 3)}')
d = []
for i in range(len(K1)):
    for j in range(len(K2)):
        y1 = sqrt(sum([(i - j) ** 2 for i,j in zip(K1[i], K2[j])]))
        d.append(y1)
print(f'Метод полной связи: {max(d)}, метод одиночной связи: {min(d)}')
LenC1C2 = abs(C1[0] * C2[0] + C1[1] * C2[1])
print(f'Метод средней связи: {sum([d]) / LenC1C2} ')

X1 = [1, 2]
X2 = [0, 3]
X3 = [-1, 1]
X4 = [0, -4]
X5 = [4, -2]
X = [X1, X2, X3, X4, X5]
D = []
d = []
for i in range(len(X)):
    D = []
    for j in range(len(X)):
        d1 = sqrt(sum([(i - j) ** 2 for i,j in zip(X[i], X[j])]))
        D.append(d1)
    D = [round(i, 3) for i in D]
    d.append(D)

print(d)


