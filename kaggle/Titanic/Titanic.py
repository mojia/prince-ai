import numpy as np
import pandas as pd
import os
import time
# from sklearn import cross_validation
# from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import math
import hashlib
from xgboost import XGBClassifier
import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import layers
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalanceCascade
from collections import Counter

dir = '/Users/xinwang/ai/dataset/kaggle/titanic/'
train_file = 'train.csv'
test_file = 'test.csv'
pd.set_option('display.width', 1300)
seed = 1000
test_size = 0.3333


def hash_func(s, n_bins=1000):
    s = s.encode('utf-8')
    return int(hashlib.md5(s).hexdigest(), 16) % (n_bins - 1) + 1


def processName(x, df):
    print(x + ' wangxin')


def standardFare(x, fares):
    return (x - fares.min()) / (fares.max() - fares.min())


def getTrainFeaturesName():
    return ['pclass_1', 'pclass_2', 'pclass_3',
            # 'Fare',
            # 'Name',
            'Age',
            'female', 'male',
            'embarked_C', 'embarked_Q', 'embarked_S',
            'cabin_A', 'cabin_B', 'cabin_C', 'cabin_D', 'cabin_E', 'cabin_F', 'cabin_G', 'cabin_n',
            "parch_0", "parch_1", "parch_2", "parch_3", "parch_4", "parch_5", "parch_6",
            # "Ticket",
            "sibsp_0", "sibsp_1", "sibsp_2", "sibsp_3", "sibsp_4", "sibsp_5", "sibsp_8"]


def processFeature(df):
    df['Name'] = df['Name'].map(lambda x: None if x is None else hash_func(x))

    p_class_onehot = pd.get_dummies(df['Pclass'], prefix='pclass')
    df = df.drop(columns=['Pclass'])
    df = df.join(p_class_onehot)

    df['Age'] = df['Age'].map(lambda x: None if math.isnan(x) else int(x / 100))

    sex_onehot = pd.get_dummies(df['Sex'])
    df = df.drop(columns=['Sex'])
    df = df.join(sex_onehot)

    sibsp_onehot = pd.get_dummies(df['SibSp'], prefix='sibsp')
    df = df.drop(columns=['SibSp'])
    df = df.join(sibsp_onehot)

    parch_onehot = pd.get_dummies(df['Parch'], prefix='parch')
    df = df.drop(columns=['Parch'])
    df = df.join(parch_onehot)

    df['Ticket'] = df['Ticket'].map(lambda x: None if x is None else hash_func(x))

    df['Fare'] = df['Fare'].map(lambda x: None if math.isnan(x) else standardFare(x, df['Fare']))

    df['Cabin'] = df['Cabin'].map(lambda x: None if x is None else str(x)[:1])
    cabin_onehot = pd.get_dummies(df['Cabin'], prefix='cabin')
    df = df.drop(columns=['Cabin'])
    df = df.join(cabin_onehot)

    embarked_onehot = pd.get_dummies(df['Embarked'], prefix="embarked")
    df = df.drop(columns=['Embarked'])
    df = df.join(embarked_onehot)

    featuresName = getTrainFeaturesName()
    x = df[featuresName]
    x = x.fillna(x.mean().astype(int))

    return x


def buildTrainData():
    df = pd.read_csv(os.path.join(dir, train_file))
    y = df['Survived'].values
    x = processFeature(df)

    x_resampled, y_resampled = SMOTE().fit_sample(x, y)
    print('x' + str(x.shape))
    print('y' + str(y.shape))
    print('x_resampled' + str(x_resampled.shape))
    print('y_resampled' + str(y_resampled.shape))
    return x_resampled, y_resampled


def buildTestX():
    df = pd.read_csv(os.path.join(dir, test_file))
    x = processFeature(df)

    return x

# -----------------------------------SVM-------------------------------------------------------


def cv_train_svm():
    x, y = buildTrainData()
    clf = svm.SVC(kernel='linear', C=3)
    scores = cross_val_score(clf, x, y, cv=5, scoring='f1_weighted', verbose=2)
    print(scores)
    print(scores.mean())

    # clf.fit(x, y)
    # test_x = buildTestX()
    # print(clf.predict(test_x))


# cv_train_svm()


def grid_search_cv_train_svm():
    x, y = buildTrainData()
    svc = svm.SVC()
    params = [
        {
            'C': [0.01, 0.1, 1, 3, 7, 9],
            'gamma':[0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
            'kernel':['rbf']
        }, {
            'C': [0.001, 0.01, 0.1, 1, 3, 7, 9],
            'kernel':['linear']
        }
    ]
    clf = GridSearchCV(svc, params, cv=5, n_jobs=8, scoring='accuracy', verbose=2)
    clf.fit(x, y)
    print(clf.best_params_)
    best_model = clf.best_estimator_


# grid_search_cv_train_svm()

# -----------------------------------xbgboost-------------------------------------------------------


def xgboosting_grid_search():
    x, y = buildTrainData()

    learning_rate = np.linspace(0.001, 0.5, 10)
    gamma = np.linspace(0.01, 1, 10)
    max_depth = range(3, 10, 2)
    subsample = [0.8]
    colsample_bytree = [0.5]
    min_child_weight = range(3, 10)
    n_estimators = range(10, 100, 8)

    param_grid = dict(learning_rate=learning_rate, max_depth=max_depth, subsample=subsample, colsample_bytree=colsample_bytree, min_child_weight=min_child_weight, gamma=gamma, n_estimators=n_estimators)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    model = XGBClassifier()
    grid_search = GridSearchCV(model, param_grid, scoring='accuracy', n_jobs=32, cv=kfold, verbose=2)
    grid_result = grid_search.fit(x, y)
    print("Best: %f , \n\nusing %s, \n\n\nscores %s" % (grid_result.best_score_, grid_result.best_params_, grid_result.grid_scores_))


# xgboosting_grid_search()


def xgboosting_cv():
    model = XGBClassifier(max_depth=5, learning_rate=0.111889, gamma=0.23, subsample=0.8, colsample_bytree=0.5, min_child_weight=4, n_estimators=26, silent=False, objective='binary:logistic')

    x, y = buildTrainData()
    scores = cross_val_score(model, x, y, cv=5, scoring='accuracy', verbose=2)

    print(scores)
    print(scores.mean())


xgboosting_cv()


def xgboosting_train():
    model = XGBClassifier(max_depth=5, learning_rate=0.111889, gamma=0.23, subsample=0.8, colsample_bytree=0.5, min_child_weight=4, n_estimators=26, silent=False, objective='binary:logistic')

    x, y = buildTrainData()
    model.fit(x, y, eval_metric='rmse')

    test_x = buildTestX()
    y_pred = model.predict(test_x)

    df = pd.read_csv(os.path.join(dir, test_file))
    print('-----------------------------------result---------------------------------------------')

    result = ''
    for i in range(418):
        print(str(df['PassengerId'].values[i]) + ',' + str(np.array(y_pred)[i]))
        pass

    return model


# xgboosting_train()

# -----------------------------------xbgboost + LR-------------------------------------------------------


def xgboosting_LR_cv():
    best_model = xgboosting_train()
    x, y = buildTrainData()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)

    xgb_x_train = best_model.apply(x_train)
    xgb_x_evaluate = best_model.apply(x_test)
    print('xgb_x_train ' + str(xgb_x_train))
    print('xgb_x_evaluate ' + str(xgb_x_evaluate.shape))

    # all features
    x_train_mix = np.hstack([xgb_x_train, x_train])
    x_evaluate_mix = np.hstack([xgb_x_evaluate, x_test])

    C_params = np.linspace(0.001, 0.5, 10)
    accuracyArray = []
    tic = time.time()

    for param in C_params:
        model = LogisticRegression(C=param, penalty='l2', max_iter=300)
        scores = cross_val_score(model, x_train_mix, y_train, cv=5, scoring='accuracy')
        accuracyArray.append(scores.mean())

    print("mean_accuracy,", accuracyArray)
    best_index = accuracyArray.index(max(accuracyArray))
    print('best param ', C_params[best_index])
    print('best_accuracy=%g ' % max(accuracyArray))

    LR = LogisticRegression(C=C_params[best_index], penalty='l2', max_iter=500)
    xgboost_x = best_model.apply(x)
    xgboost_x_mix = np.hstack([xgboost_x, x])
    scores = cross_val_score(LR, xgboost_x_mix, y, cv=5, scoring='accuracy', verbose=2)

    print(scores)
    print(scores.mean())


# xgboosting_LR_cv()


def xgboosting_LR_train():
    grd_enc = OneHotEncoder()
    grd_lm = LogisticRegression()

    best_model = xgboosting_train()
    x, y = buildTrainData()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)

    xgb_x_train = best_model.apply(x_train)
    xgb_x_evaluate = best_model.apply(x_test)
    print('xgb_x_train ' + str(xgb_x_train))
    print('xgb_x_evaluate ' + str(xgb_x_evaluate.shape))

    # all features
    x_train_mix = np.hstack([xgb_x_train, x_train])
    x_evaluate_mix = np.hstack([xgb_x_evaluate, x_test])

    C_params = np.linspace(0.001, 0.5, 10)
    LR_aucs = []
    tic = time.time()

    for param in C_params:
        model = LogisticRegression(C=param, penalty='l2', max_iter=300)
        scores = cross_val_score(model, x_train_mix, y_train, cv=5, scoring='roc_auc')
        LR_aucs.append(scores.mean())

    print("mean_auc,", LR_aucs)
    best_index = LR_aucs.index(max(LR_aucs))
    print('best param ', C_params[best_index])
    print('best_auc=%g ' % max(LR_aucs))

    LR = LogisticRegression(C=C_params[best_index], penalty='l2', max_iter=100)
    LR.fit(x_train_mix, y_train)

    y_pred_prob = LR.predict_proba(x_evaluate_mix)[:1]
    print(y_pred_prob[:10])

    test_x = buildTestX(isTest=True)
    xgb_test_x = best_model.apply(test_x)
    test_x_mix = np.hstack([xgb_test_x, test_x])
    test_pred = LR.predict(test_x_mix)
    print('-----------------------------------test_pred---------------------------------------------')
    print(test_pred)

    df = pd.read_csv(os.path.join(dir, test_file))
    print('-----------------------------------result---------------------------------------------')

    result = ''
    for i in range(418):
        print(str(df['PassengerId'].values[i]) + ',' + str(np.array(test_pred)[i]))


# xgboosting_LR_train()

# -----------------------------------Keras-------------------------------------------------------


def createSequentialModel():
    model = Sequential()

    model.add(layers.Dense(200, name='hidden_layer_1', activation='tanh', input_dim=35))
    # model.add(layers.Dropout(0.2))
    # model.add(layers.Dense(100, name='hidden_layer_2', activation=config.activation))
    # model.add(layers.Dense(8, name='hidden_layer_3', activation='tanh'))
    # model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, name='output_layer', activation='sigmoid'))

    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    sgd = optimizers.SGD(lr=0.01, momentum=0.0, decay=0, nesterov=False)
    adagrad = optimizers.Adagrad(lr=0.01, epsilon=1e-06)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model


def create_baseline():
    # create model
    model = Sequential()
    model.add(layers.Dense(50, input_dim=31, kernel_initializer='normal', activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(50,  activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(50,  activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(50, activation='relu'))

    model.add(layers.Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model. We use the the logarithmic loss function, and the Adam gradient optimizer.
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def keras_train():
    x, y = buildTrainData()
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_y = encoder.transform(y)
    seed = 1001
    # print(encoded_y)

    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=200, batch_size=5, verbose=2)))
    pipeline = Pipeline(estimators)

    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    results = cross_val_score(pipeline, x, encoded_y, cv=kfold, scoring='accuracy')
    print("Results: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))


# keras_train()
