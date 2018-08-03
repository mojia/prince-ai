import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from FeatureCenter import FeatureCenter


featureCenter = FeatureCenter(
    "SH600000", "2001-01-09 09:35:00", "2018-10-23 09:35:00")

train_x, train_y, evaluate_x, evaluate_y = featureCenter.getDataSet()
print('train_x.shape:' + str(train_x.shape) +
      ' train_y.shape:' + str(train_y))
print('evaluate_x.shape:' + str(evaluate_x.shape) +
      ' evaluate_y.shape:' + str(evaluate_y))

clf = DecisionTreeClassifier()
clf.fit(train_x, train_y)

evaluate_predict = clf.predict(evaluate_x)
print(classification_report(evaluate_y, evaluate_predict))
