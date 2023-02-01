from numpy import *


def dist1(x1, x2):
    de = sqrt(sum([(i - j) ** 2 for i, j in zip(x1,x2)]))
    dm = sum([abs(i - j) for i, j in zip(x1,x2)])
    dch = max([i - j for i, j in zip(x1,x2)])
    print('Евклидово расстояние =',round(de,3))
    print('Манхеттенское расстояние =',round(dm,3))
    print('Расстояние Чебышёва =',round(dch,3))


def dist2(x1,x2):
    de = sqrt(sum([(i - j) ** 2 for i, j in zip(x1,x2)]))
    dm = sum([abs(i - j) for i, j in zip(x1,x2)])
    dch = max([i - j for i, j in zip(x1,x2)])
    return de

def dist3(x1,x2):
    de = sqrt(sum([(i - j) ** 2 for i, j in zip(x1,x2)]))
    dm = sum([abs(i - j) for i, j in zip(x1,x2)])
    dch = max([i - j for i, j in zip(x1,x2)])
    return 1 / de ** 2

T = array([52,18])
X = array([[31,19], [45,23], [15,46], [92,82], [78,29], [58,34],[25,19],[29,93],[84,82],[82,27]])
dev = []
dma = []
dch = []
for i in range(len(X)):
    m = dist2(T,X[i])
    dev.append(m)
print(dev)
print('Класс "0":',round(dev[2] + dev[4],3),'Класс "1":', round(dev[0] + dev[1] + dev[3],4))









