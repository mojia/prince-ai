from Config import Config
import os
from DBHelper import DBHelper
import numpy as np
from FigurePoints import FigurePoints
import matplotlib.pyplot as plt
from CSVFileUtil import CSVFileUtil
from keras import utils

config = Config()


class FeatureCenter:
    def __init__(self, code, startDateTime, endDateTime):
        self.code = code
        self.startDateTime = startDateTime
        self.endDateTime = endDateTime
        self.dbHelper = DBHelper()
        self.dataDim = config.getDataDim()
        self.debugTrainData = config.debugForPrepareData
        self.backWindow = config.backWindowLength

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

        index = i - self.backWindow
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

    def buildSample(self, i, kLines):
        sample = self.buildPriceFeature(i, kLines)

        if config.addBarFeatures is True:
            sample.extend(self.buildBARFeature(i, kLines))

        if self.debugTrainData is True:
            print('sample\n' + str(sample))

        sample.extend([kLines[i].label])
        sampleArray = np.array(sample).reshape((1, self.dataDim))

        return sampleArray

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
        priceDim = config.backWindowLength
        if config.debugForPrepareData is True:
            print('norm data count:' + str(count))
            print('norm data priceDim:' + str(priceDim))

        priceNormVec = self.meanNorm(data[:, :priceDim])

        if config.addBarFeatures is True:
            barNormVec = self.meanNorm(data[:, priceDim: 2 * priceDim])
            standard_x = np.hstack((priceNormVec, barNormVec))
        else:
            standard_x = np.array(priceNormVec)

        y = data[:, -1]
        standard_y = utils.to_categorical(np.array(y), num_classes=3)

        return standard_x, standard_y

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

    def drawX(self, x, y):
        if config.debugForPrepareData is True:
            x = np.array(x)
            print('drawX\n' + str(x))
            drawData = []
            for i in range(len(x)):
                label = np.argmax(y[i])
                drawData.append(
                    FigurePoints(x[i][:config.backWindowLength],
                                 str(i) + str(' ') + str(label)))
                # drawData.append(FigurePoints(
                #     x[i][config.backWindowLength:2 * config.backWindowLength],
                #     "DEA"))
            self.drawMultiLines(drawData)

    def buildInputData(self, kLines):
        count = int(len(kLines) * 0.8)
        if config.skipStep == 0:
            input_kLines_train = kLines[:count]
            input_kLines_evaluate = kLines[count:]
        else:
            input_train = kLines[:count]
            input_evaluate = kLines[count:]

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

    def discard(self, c0, c1, c2, cLabel):
        minSize = config.minSizeSamples
        if c0 < minSize or c1 < minSize or c2 < minSize:
            return False

        c0p = c0 * 100.0 / (c0 + c1 + c2)
        c1p = c1 * 100.0 / (c0 + c1 + c2)
        c2p = c2 * 100.0 / (c0 + c1 + c2)

        array = [c0p, c1p, c2p]
        argmax = np.argmax(np.array(array))
        argmin = np.argmin(np.array(array))

        if array[argmax] - array[argmin] > 10:
            return True
        else:
            return False

    def buildXY(self, kLines):
        xy = np.zeros((0, self.dataDim), dtype='float')
        print('--------buildX----------')
        print('kLines size ' + str(len(kLines)))
        print('in xy.shape ' + str(xy.shape))

        countLabel0 = 0
        countLabel1 = 0
        countLabel2 = 0
        for i in range(len(kLines)):
            if i >= config.backWindowLength:
                sample = self.buildSample(i, kLines)
                if kLines[i].label == 0:
                    countLabel0 += 1
                elif kLines[i].label == 1:
                    countLabel1 += 1
                elif kLines[i].label == 2:
                    countLabel2 += 1
                if self.discard(countLabel0, countLabel1,
                                countLabel2, kLines[i].label):
                    pass
                else:
                    xy = np.vstack((sample, xy))

                if i % 1000 == 0:
                    print('buildX complete ' + (str(i)) + " " +
                          str(i * 100.0 / len(kLines)) + " " +
                          str(countLabel0) + " " +
                          str(countLabel1) + " " +
                          str(countLabel2) + " ")

                if self.debug() is True:
                    if i == config.backWindowLength + config.debugSampleCount:
                        break

            i += 1

        print('xy.shape ' + str(xy.shape))
        # self.drawX(xy[:self.dataDim - 1], 'xdata')
        standard_x, standard_y = self.norm(xy)
        self.drawX(standard_x, standard_y)

        return standard_x, standard_y

    def buildTrainAndEvaluateData(self):
        kLines = self.dbHelper.query(
            self.code, self.startDateTime, self.endDateTime)
        print(len(kLines))

        input_kLines_train, input_kLines_evaluate = \
            self.buildInputData(kLines)

        train_x, train_y = self.buildXY(input_kLines_train)
        evaluate_x, evaluate_y = self.buildXY(input_kLines_evaluate)

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
