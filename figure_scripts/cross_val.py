import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


data = pd.read_csv('training_data.csv')
X_train, X_test, y_train, y_test = train_test_split(data.loc[:, data.columns != ' Class'], data[' Class'], test_size=0.95, random_state=0)


parameters = {'kernel':('linear', 'rbf'), 'C':[0.1, .5, 1, 5, 10]}



svc = svm.SVC(gamma="scale")
clf = GridSearchCV(svc, parameters, cv=5)
clf.fit(X_train, y_train)

sorted(clf.cv_results_.keys())

clf.score(X_test, y_test)

