from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
import cv2
from imutils import paths
import os
from numpy import zeros, ones, concatenate, vstack, round


os.chdir(r'C:\Users\Пользователь\Documents\Обучение\Магистратура\1 семестр\Введение в МО\Ensemble methods\train')
print("Current working directory is:", os.getcwd())


def extract_histogram(image, bins=(8, 8, 8)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


imagePaths = sorted(list(paths.list_images('train')))
X = zeros(512)
Y1 = ones(500)
Y2 = zeros(500)
Y = concatenate((Y1, Y2))

for file in os.listdir(r'C:\Users\Пользователь\Documents\Обучение\Магистратура\1 семестр\Введение в МО\Ensemble methods\train'):
    if file[file.rfind('.') + 1:] in ['jpg']:
        image = cv2.imread(file)
        new_row = extract_histogram(image)
        X = vstack([X, new_row])

X = X[1:, :]
X_train = X
Y_train = Y


# Построение моделей
svm = LinearSVC(C=1.51, random_state=336).fit(X_train, Y_train)
print(svm.score(X_train, Y_train))
bag = BaggingClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy', min_samples_leaf=10,
                                                              max_leaf_nodes=20, random_state=336), n_estimators=21,
                        random_state=336).fit(X_train, Y_train)
print(bag.score(X_train, Y_train))
rf = RandomForestClassifier(n_estimators=21, criterion='entropy', min_samples_leaf=10, max_leaf_nodes=20,
                            random_state=336).fit(X_train, Y_train)
print(rf.score(X_train, Y_train))
estimators = [('SVM', svm), ('Bagging', bag), ('RandFor', rf)]
# Обучение метаалгоритма
clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(solver='lbfgs', random_state=336),cv=2)
clf.fit(X_train, Y_train)

ma = clf.score(X_train, Y_train)
print(f'Accuracy: {round(ma, 3)}')

os.chdir(r'C:\Users\Пользователь\Documents\Обучение\Магистратура\1 семестр\Введение в МО\Ensemble methods\test')
print("Current working directory is:", os.getcwd())

imagePaths = sorted(list(paths.list_images('test')))
X1 = zeros(512)
Y11 = zeros(50)
Y22 = ones(50)
Yf = concatenate((Y11, Y22))
for file in os.listdir(r'C:\Users\Пользователь\Documents\Обучение\Магистратура\1 семестр\Введение в МО\Ensemble methods\test'):
    if file[file.rfind('.') + 1:] in ['jpg']:
        image = cv2.imread(file)
        new_row = extract_histogram(image)
        X1 = vstack([X1, new_row])

X1 = X1[1:, :]
X_test = X1
Y_test = Yf

image1 = cv2.imread('cat.1012.jpg')
image2 = cv2.imread('cat.1022.jpg')
image3 = cv2.imread('dog.1021.jpg')
image4 = cv2.imread('dog.1038.jpg')

hst1 = extract_histogram(image1)
hst2 = extract_histogram(image2)
hst3 = extract_histogram(image3)
hst4 = extract_histogram(image4)

print(f'Предсказание для картинки cat.1012.jpg = {round(clf.predict_proba(hst1.reshape(1, -1)),3)}')
print(f'Предсказание для картинки cat.1022.jpg = {round(clf.predict_proba(hst2.reshape(1, -1)),3)}')
print(f'Предсказание для картинки dog.1021.jpg = {round(clf.predict_proba(hst3.reshape(1, -1)),3)}')
print(f'Предсказание для картинки dog.1038.jpg = {round(clf.predict_proba(hst4.reshape(1, -1)),3)}')
