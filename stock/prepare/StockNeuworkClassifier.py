import numpy as np
from keras.models import Sequential
from keras import layers
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from DataCenter import DataCenter

import matplotlib.pyplot as plt
from Config import Config
from FeatureCenter import FeatureCenter
from PerformanceModel import PerformanceModel
from PerformanceDBHelper import PerformanceDBHelper

config = Config()
output_dim = 3
dataDim = config.getDataDim()
backWindow = config.backWindowLength
debugTrainData = config.debugForPrepareData
seed = 7
np.random.seed(seed)


class StockNeuworkClassifier:
    def __init__(self, codes, startDateTime, endDateTime):
        self.codes = codes
        self.startDateTime = startDateTime
        self.endDateTime = endDateTime
        performanceDBHelper = PerformanceDBHelper()
        self.dataCenter = DataCenter()

        pass

    def createModel(self, perfModel):
        model = Sequential()

        model.add(layers.Dense(config.hidden_layer_1_unit, name='hidden_layer_1',
                               activation=config.activation, input_dim=dataDim - 1))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(100, name='hidden_layer_2', activation=config.activation))
        # model.add(layers.Dense(8, name='hidden_layer_3', activation='tanh'))
        # model.add(layers.Dropout(0.2))
        model.add(layers.Dense(output_dim, name='output_layer', activation='softmax'))

        adam = optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        sgd = optimizers.SGD(lr=0.01, momentum=0.0, decay=0, nesterov=False)
        adagrad = optimizers.Adagrad(lr=0.01, epsilon=1e-06)
        loss = 'categorical_crossentropy'
        model.compile(loss=loss, optimizer=sgd, metrics=['accuracy'])

        perfModel.setModelFields(config.hidden_layer_1_unit, config.epochs,
                                 config.activation, sgd, loss, config.kFold)

        return model

    def drawTrainData(self, x, label='', plot=True, color='red'):
        fig = plt.figure(figsize=(30, 8))
        if plot is True:
            plt.plot(range(len(x)), x, color=color)
        plt.scatter(range(len(x)), x)
        plt.ylabel('Label:' + str(label))

    def tryNetwork(self, X, Y, evaluate_x, evaluate_y):
        perfModel = PerformanceModel(str(X.shape))
        perfModel.setDataFields(self.codes, self.startDateTime, self.endDateTime, config.backWindowLength,
                                config.futureWindow, config.skipStep, config.minSizeSamples)

        model = self.createModel(perfModel)

        model.fit(X, Y, epochs=config.epochs)

        print("#####evaluate####")
        lose, accuracy = model.evaluate(evaluate_x, evaluate_y)
        print("\n\nAccuracy:" + str(accuracy * 100))

        # return ,accuracy * 100

    def tryCV(self, X, Y):
        estimator = KerasClassifier(build_fn=self.createModel, epochs=config.epochs, batch_size=32, verbose=1)
        kfold = KFold(n_splits=config.kFold, shuffle=True, random_state=seed)
        result = cross_val_score(estimator, X, Y, cv=kfold)
        print('baseLine: %.2f%% (%.2f%%)' % (result.mean() * 100, result.std() * 100))

    def trainAndTest(self):
        X, Y, evaluate_x, evaluate_y = self.dataCenter.loadData(self.codes, self.startDateTime, self.endDateTime)

        # self.tryCV(X, Y)
        if config.debugForPrepareData is False:
            self.tryNetwork(X, Y, evaluate_x, evaluate_y)


if __name__ == '__main__':
    StockNeuworkClassifier(
        ["SH600000", "SH600004", "SH600005", "SH600006", "SH600007", "SH600008"], "2001-01-09 09:35:00", "2017-10-23 09:35:00").trainAndTest()
    plt.show()
