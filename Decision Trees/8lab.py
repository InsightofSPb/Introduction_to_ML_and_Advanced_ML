from math import log
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


P = [0.5, 0.15, 0.35]
omega = 0
for i in range(0, 3):
    omega -= P[i] * log(P[i], 2)

P = [0.5, 1 / 6, 1 / 3]
G = 0
for i in range(0, len(P)):
    G += P[i] * (1 - P[i])

# Main part
Source = pd.read_csv('diabetes.csv')
X = Source.head(670)
Y0 = X[X['Outcome'] == 0]
Y = X.loc[:, "Outcome"]
X = X.drop('Outcome', axis=1)
print(f'Число не больных диабетом в первых 670 = {Y0.shape[0]}')

X_train = X.head(536).to_numpy()
Y_train = Y.head(536).to_numpy()
X_test = X.tail(134).to_numpy()
Y_test = Y.tail(134).to_numpy()
clf = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=10, max_leaf_nodes=10,
                                 random_state=2020).fit(X_train, Y_train)
feature_names = Source.drop(['Outcome'], axis=1).columns.values.tolist()
class_names = ['0', '1']
fig = plt.figure(figsize=(20, 20))
plot = tree.plot_tree(clf, feature_names=feature_names, class_names=class_names, filled=True)
print(f'Глубина дерева = {clf.get_depth()}')

# plt.show()


matrix = confusion_matrix(clf.predict(X_test), Y_test)
Pr1 = matrix[1, 1] / (matrix[1, 1] + matrix[0, 1])
Pr2 = matrix[0, 0] / (matrix[0, 0] + matrix[1, 0])
R1 = matrix[1, 1] / (matrix[1, 1] + matrix[1, 0])
R2 = matrix[0, 0] / (matrix[0, 0] + matrix[0, 1])
Macro_F1 = round((2 * Pr1 * R1 / (Pr1 + R1) + 2 * Pr2 * R2 / (Pr2 + R2)) / 2, 2)
print(len(Y_test))
print(f'Доля верных ответов = {round((matrix[1, 1] + matrix[0, 0]) / len(Y_test),2)}')
print(f'Macro_F1= {Macro_F1}')

Numb = [712, 749, 703, 740]
for i in range(0, len(Numb)):
    print(f' Пациент номер {Numb[i]} имеет класс =  {clf.predict(Source.loc[Numb[i], :"Age"].values.reshape(1, -1))}')
