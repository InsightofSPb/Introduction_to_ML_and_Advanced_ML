## Задание
В данной задаче вам предстоит решить задачу классификации изображений – отделить изображения кошек от изображений собак, используя классификацию с мягким зазором и параметром <i>C</i>. В архиве train_task.zip находится выборка, включающая в себя изображения кошек и собак (по 500 изображений). Имя каждого изображения, для удобства, имеет следующий формат:  cat/dog.номер_изображения.jpg  в зависимости от того, какое животное присутствует на изображении. Данная выборка используется для обучения классификатора и оценки классификатора.
Для работы с изображениями и получения их гистограмм — характеристик распределения интенсивности изображения, следует воспользоваться следующей функцией и библиотекой cv2.

#### Подключаем все необходимые модули и библиотеки
```python
import cv2
from sklearn.svm import LinearSVC
from imutils import paths
import os
from numpy import zeros, ones, concatenate, vstack, round
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
```
#### Создаём функцию для получения характеристик распределения интенсивности изображений
```python
def extract_histogram(image, bins=(8, 8, 8)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()
```
#### Отсортируем данные и закодируем котов и собак 0 и 1 соответственно
```python
imagePaths = sorted(list(paths.list_images('train')))
X = zeros(512)
Y1 = zeros(500)
Y2 = ones(500)
Y = concatenate((Y1, Y2))
```
#### Применим функцию extract_histogram, с помощью строкового метода rfind будем находить нужные нам файлы
```python
for file in os.listdir(r'C:\Users\Пользователь\Documents\Обучение\Магистратура\1 семестр\Введение в МО\Support Vector Machines SVM\train'):
    if file[file.rfind('.') + 1:] in ['jpg']:
        image = cv2.imread(file)
        new_row = extract_histogram(image)
        X = vstack([X,new_row])  # объединим массивы
X = X[1:, :]
```

#### Разделим данные на тренировочные и тестовые и обучим модель, вычислим параметры (в соответствии с заданием)
```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25 , random_state=13)

w = LinearSVC(C = 1.28, random_state= 13).fit(X_train, Y_train)

theta = [1, 34, 335]
print(f'theta{theta[0]} = {round(w.coef_[0][0], 2)}')
print(f'theta{theta[1]} = {round(w.coef_[0][33], 2)}')
print(f'theta{theta[2]} = {round(w.coef_[0][334], 2)}')
```
#### Составим матрицу ошибок, рассчитаем точность и полноту, вычислим F_1 метрику
```python
matrix = confusion_matrix(w.predict(X_test), Y_test.reshape(-1, 1))

Pr1 = matrix[1, 1] / (matrix[1, 1] + matrix[0, 1])
Pr2 = matrix[0, 0] / (matrix[0, 0] + matrix[1, 0])
R1 = matrix[1, 1] / (matrix[1, 1] + matrix[1, 0])
R2 = matrix[0, 0] / (matrix[0, 0] + matrix[0, 1])
Macro_F1 = round((2 * Pr1 * R1 / (Pr1 + R1) + 2 * Pr2 * R2 / (Pr2 + R2)) / 2, 2)
print(f'Macro_F1= {Macro_F1}')
```
#### Для новых картинок из нового набора данных найдём класс, к которому они относятся
```python
os.chdir(r'C:\Users\Пользователь\Documents\Обучение\Магистратура\1 семестр\Введение в МО\Support Vector Machines SVM\test')
print("Current working directory is:", os.getcwd())

image1 = cv2.imread('cat.1043.jpg')
image2 = cv2.imread('cat.1037.jpg')
image3 = cv2.imread('dog.1012.jpg')
image4 = cv2.imread('dog.1033.jpg')

hst1 = extract_histogram(image1)
hst2 = extract_histogram(image2)
hst3 = extract_histogram(image3)
hst4 = extract_histogram(image4)

print(f'Предсказание для картинки cat.1043.jpg = {w.predict(hst1.reshape(1, -1))}')
print(f'Предсказание для картинки cat.1037.jpg = {w.predict(hst2.reshape(1, -1))}')
print(f'Предсказание для картинки dog.1012.jpg = {w.predict(hst3.reshape(1, -1))}')
print(f'Предсказание для картинки dog.1033.jpg = {w.predict(hst4.reshape(1, -1))}')
```