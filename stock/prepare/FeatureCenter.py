from config import config
import os
from DBHelper import DBHelper
import numpy as np
from FigurePoints import FigurePoints
import matplotlib.pyplot as plt
from CSVFileUtil import CSVFileUtil
from keras import utils


class FeatureCenter:
    def __init__(self, code, startDateTime, endDateTime):
        self.code = code
        self.startDateTime = startDateTime
        self.endDateTime = endDateTime
        self.dbHelper = DBHelper()
        self.features_dim = config.featureDim
        self.debugTrainData = config.debugForPrepareData
        self.backWindow = config.backWindowLength

    def buildY(self, kLines):
        y = [(k.label if k.label is not None else 1)
             for k in kLines]
        y = y[self.backWindow:]
        y = utils.to_categorical(np.array(y), num_classes=3)

        return y

    def buildBARFeature(self, i, kLines):
        featureVec = []

        index = i - self.backWindow
        while len(featureVec) < self.backWindow:
            featureVec.append(kLines[index].bar)
            index += 1

        return featureVec

    def buildDEAFeature(self, i, kLines):
        featureVec = []

        index = i - self.backWindow
        while len(featureVec) < self.backWindow:
            featureVec.append(kLines[index].dea)
            index += 1

        return featureVec

    def buildDIFFeature(self, i, kLines):
        featureVec = []

        index = i - backWindow
        while len(featureVec) < self.backWindow:
            featureVec.append(kLines[index].dif)
            index += 1

        return featureVec

    def buildPriceFeature(self, i, kLines):
        featureVec = []

        index = i - self.backWindow
        while len(featureVec) < self.backWindow:
            featureVec.append(kLines[index].closePrice)
            index += 1

        return featureVec

    def buildFeature(self, i, kLines):
        featureVec = []

        featureVec.append(self.buildPriceFeature(i, kLines))
        # featureVec.append(buildDIFFeature(i, kLines))
        # featureVec.append(buildDEAFeature(i, kLines))
        featureVec.append(self.buildBARFeature(i, kLines))

        feature = np.array(featureVec).reshape((1, self.features_dim))

        return feature

    def debug(self):
        return config.debugForPrepareData

    def meanNorm(self, linearData):
        if self.debugTrainData is True:
            print('-----------meanNorm---------')
        min, mean, max = linearData.min(), linearData.mean(), linearData.max()

        linearData = (linearData - mean) * 0.99 / (max - min) + 0.01

        return linearData

    def maxminNorm(self, linearData):
        if self.debugTrainData is True:
            print('-----------maxminNorm---------')
            print(linearData)
        min, max = linearData.min(), linearData.max()

        linearData = (linearData - min) * 0.99 / (max - min) + 0.01

        return linearData

    def positiveNegativeNorm(self, linearData):
        if self.debugTrainData is True:
            print('-----------maxminNorm---------')

        data = []
        for i in range(len(linearData)):
            x = linearData[i]
            x = [0.99 if item >= 0 else 0.01 for item in x]
            data.append(x)

        return np.array(data)

    def norm(self, data):
        count = len(data)

        # price norm
        priceDim = int(len(data.T) / 2)
        if config.debugForPrepareData is True:
            print('norm data count:' + str(count))
            print('norm data priceDim:' + str(priceDim))

        priceNormVec = self.maxminNorm(data[:, :priceDim])
        # deaNormVec = meanNorm(data[:, priceDim:])
        barNormVec = self.positiveNegativeNorm(data[:, priceDim:])

        array_standard = np.hstack((priceNormVec, barNormVec))

        return array_standard

        # difNormVec = meanNorm(data[:, backWindow: 2 * backWindow])

    def drawMultiLines(self, figureDatas):
        fig = plt.figure(figsize=(30, 8))
        maxColums = 4
        maxRows = int(len(figureDatas) / maxColums) + 1

        index = 0
        subplotShape = (maxRows, maxColums)
        for row in range(maxRows):
            for colum in range(maxColums):
                figurePoints = figureDatas[index]
                plt.subplot2grid(subplotShape, (row, colum))
                plt.plot(range(len(figurePoints.x)), figurePoints.x)
                plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
                plt.scatter(range(len(figurePoints.x)), figurePoints.x)
                plt.title(str(figurePoints.title))
                index += 1

                if index >= len(figureDatas):
                    return

    def drawX(self, x, label):
        if config.debugForPrepareData is True:
            drawData = []
            for i in range(4):
                drawData.append(
                    FigurePoints(x[i][:config.backWindowLength],
                                 label + str(' ') + str(i) + " price"))
                drawData.append(FigurePoints(
                    x[i][config.backWindowLength:2 * config.backWindowLength],
                    "DEA"))
            self.drawMultiLines(drawData)

    def buildInputData(self, kLines):
        if config.skipStep == 0:
            input_kLines_train = kLines
            input_kLines_evaluate = kLines[:config.evaluateCount]
        else:
            input_train = kLines
            input_evaluate = kLines[:config.evaluateCount]

            input_kLines_train = []
            input_kLines_evaluate = []
            index = 0
            step = config.skipStep
            while (index < len(input_train)):
                input_kLines_train.append(input_train[index])
                index += step

            index = 0
            while (index < len(input_evaluate)):
                input_kLines_evaluate.append(input_evaluate[index])
                index += step

        print('input_kLines_train size:' + str(len(input_kLines_train)))
        print('input_kLines_evaluate size:' + str(len(input_kLines_evaluate)))

        return input_kLines_train, input_kLines_evaluate

    def buildX(self, kLines):
        x = np.zeros((0, self.features_dim), dtype='float')
        print('--------buildX----------')
        print('kLines size ' + str(len(kLines)))
        print('in x.shape ' + str(x.shape))

        for i in range(len(kLines)):
            if i >= config.backWindowLength:
                feature = self.buildFeature(i, kLines)
                x = np.vstack((feature, x))

                if i % 100 == 0:
                    print('buildX complete ' + (str(i)) +
                          str(' ') + str(i * 100.0 / len(kLines)))

                if self.debug() is True:
                    if i == config.backWindowLength + 4:
                        break

            i += 1

        self.drawX(x, 'xdata')
        featuresNorm = self.norm(x)
        self.drawX(featuresNorm, 'xdataNorm')

        print('featuresNorm\n' + str(featuresNorm))

        return featuresNorm

    def buildTrainAndEvaluateData(self):
        kLines = self.dbHelper.query(
            self.code, self.startDateTime, self.endDateTime)
        print(len(kLines))

        input_kLines_train, input_kLines_evaluate = \
            self.buildInputData(kLines)

        train_x = self.buildX(input_kLines_train)
        train_y = self.buildY(input_kLines_train)

        evaluate_x = self.buildX(input_kLines_evaluate)
        evaluate_y = self.buildY(input_kLines_evaluate)

        return np.array(train_x), train_y, \
            np.array(evaluate_x), evaluate_y

    def existDataSet(self, code):
        train_x_file = 'train_x_' + code + '.csv'
        train_y_file = 'train_y_' + code + '.csv'
        evaluate_x_file = 'evaluate_x_' + code + '.csv'
        evaluate_y_file = 'evaluate_y_' + code + '.csv'

        return CSVFileUtil.fileExist(train_x_file) and \
            CSVFileUtil.fileExist(train_y_file) and \
            CSVFileUtil.fileExist(evaluate_x_file) and \
            CSVFileUtil.fileExist(evaluate_y_file)

    def saveDataSetToFile(self, train_x, train_y,
                          evaluate_x, evaluate_y):
        train_x_file = 'train_x_' + self.code + '.csv'
        CSVFileUtil.writeCSV(train_x_file, train_x)

        train_y_file = 'train_y_' + self.code + '.csv'
        CSVFileUtil.writeCSV(train_y_file, train_y)

        evaluate_x_file = 'evaluate_x_' + self.code + '.csv'
        CSVFileUtil.writeCSV(evaluate_x_file, evaluate_x)

        evaluate_y_file = 'evaluate_y_' + self.code + '.csv'
        CSVFileUtil.writeCSV(evaluate_y_file, evaluate_y)

    def readDataSet(self):
        train_x_file = 'train_x_' + self.code + '.csv'
        train_x = CSVFileUtil.readCSV(train_x_file)

        train_y_file = 'train_y_' + self.code + '.csv'
        train_y = CSVFileUtil.readCSV(train_y_file)

        evaluate_x_file = 'evaluate_x_' + self.code + '.csv'
        evaluate_x = CSVFileUtil.readCSV(evaluate_x_file)

        evaluate_y_file = 'evaluate_y_' + self.code + '.csv'
        evaluate_y = CSVFileUtil.readCSV(evaluate_y_file)

        print('read dataset from local file')

        return train_x, train_y, evaluate_x, evaluate_y

    def getDataSet(self):
        if self.existDataSet(self.code) and self.debug() is False:
            return self.readDataSet()
        else:
            train_x, train_y, evaluate_x, evaluate_y = \
                self.buildTrainAndEvaluateData()

            if self.debug() is False:
                print('build dataset and save to file')
                self.saveDataSetToFile(train_x, train_y,
                                       evaluate_x, evaluate_y)
            return train_x, train_y, evaluate_x, evaluate_y
