import numpy as np
from keras.models import Sequential
from keras import layers
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt
from config import config
from FeatureCenter import FeatureCenter

output_dim = 3
features_dim = config.featureDim
backWindow = config.backWindowLength
debugTrainData = config.debugForPrepareData
seed = 7
np.random.seed(seed)


class StockNeuworkClassifier:
    def __init__(self):
        pass

    def createModel(self):
        model = Sequential()

        model.add(layers.Dense(config.hidden_layer_1_unit, name='hidden_layer_1',
                               activation=config.activation, input_dim=features_dim - 1))
        # model.add(layers.Dropout(0.2))
        # model.add(layers.Dense(100, name='hidden_layer_2',
        # activation='tanh'))
        # model.add(layers.Dense(8, name='hidden_layer_3', activation='tanh'))
        # model.add(layers.Dropout(0.2))
        model.add(layers.Dense(output_dim, name='output_layer', activation='softmax'))

        adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        loss = 'categorical_crossentropy'
        model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])

        return model

    def drawTrainData(self, x, label='', plot=True, color='red'):
        fig = plt.figure(figsize=(30, 8))
        if plot is True:
            plt.plot(range(len(x)), x, color=color)
        plt.scatter(range(len(x)), x)
        plt.ylabel('Label:' + str(label))

    def countLabel(self, y):
        count0 = 0
        count1 = 0
        count2 = 0
        for i in range(len(y)):
            yVector = y[i]
            index = np.argmax(yVector)
            if index == 0:
                count0 += 1
            if index == 1:
                count1 += 1
            if index == 2:
                count2 += 1
        if config.debugForPrepareData is True:
            print('count0 ' + str(count0))
            print('count1 ' + str(count1))
            print('count2 ' + str(count2))

        return count0, count1, count2

    def tryNetwork(self, X, Y, evaluate_x, evaluate_y):
        model = self.createModel()

        model.fit(X, Y, epochs=config.epochs, batch_size=32)

        print("#####evaluate####")
        lose, accuracy = model.evaluate(evaluate_x, evaluate_y, batch_size=32)
        print("\n\nAccuracy:" + str(accuracy * 100))

    def tryCV(self, X, Y):
        estimator = KerasClassifier(build_fn=self.createModel, epochs=config.epochs, batch_size=32, verbose=1)
        kfold = KFold(n_splits=config.kFold, shuffle=True, random_state=seed)
        result = cross_val_score(estimator, X, Y, cv=kfold)
        print('baseLine: %.2f%% (%.2f%%)' % (result.mean() * 100, result.std() * 100))

    def trainAndTest(self, codes, startDateTime, endDateTime):
        X = np.zeros((0, features_dim - 1), dtype='float')
        Y = np.zeros((0, 3), dtype='int')
        evaluate_x = np.zeros((0, features_dim - 1), dtype='float')
        evaluate_y = np.zeros((0, 3), dtype='int')

        for i in range(len(codes)):
            featureCenter = FeatureCenter(codes[i], startDateTime, endDateTime)

            tx, ty, ex, ey = featureCenter.getDataSet()
            X = np.vstack((X, tx))
            Y = np.vstack((Y, ty))
            evaluate_x = np.vstack((evaluate_x, ex))
            evaluate_y = np.vstack((evaluate_y, ey))

        if debugTrainData is True:
            for i in range(len(X)):
                # drawTrainData(X[i], Y[i])
                pass
        else:
            print('start training ... ')
            print('Final X.shape:' + str(X.shape))
            print('Final Y.shape:' + str(Y.shape))
            print('Final evaluate_x.shape:' + str(evaluate_x.shape))
            print('Final evaluate_y.shape:' + str(evaluate_y.shape))
            self.countLabel(Y)

            # self.tryCV(X, Y)
            self.tryNetwork(X, Y, evaluate_x, evaluate_y)


if __name__ == '__main__':
    StockNeuworkClassifier().trainAndTest(
        ["SH600000", "SH600004", "SH600005", "SH600006", "SH600007", "SH600008"], "2001-01-09 09:35:00", "2017-10-23 09:35:00")
