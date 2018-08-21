from OriginalDataSaver import OriginalDataSaver
from LabelProcessor import LabelProcessor
from MacdProcessor import MacdProcessor
import numpy as np
from Config import Config
from FeatureCenter import FeatureCenter

startDateTime = "2001-10-01 09:35:00"
endDateTime = "2020-10-20 09:35:00"
csvFileName = 'SH600021.CSV'
config = Config()


class DataCenter:
    def __init__(self):
        pass

    def profile(func):
        def wrapper(*args, **kwargs):
            import time
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print('COST: {}'.format(end - start))
            return result
        return wrapper

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

    @profile
    def loadData(self, codes, startDateTime, endDateTime):
        dataDim = config.getDataDim()
        X = np.zeros((0, dataDim - 1), dtype='float')
        Y = np.zeros((0, 3), dtype='int')
        evaluate_x = np.zeros((0, dataDim - 1), dtype='float')
        evaluate_y = np.zeros((0, 3), dtype='int')

        for i in range(len(codes)):
            featureCenter = FeatureCenter(codes[i], startDateTime, endDateTime)

            tx, ty, ex, ey = featureCenter.getDataSet()
            X = np.vstack((X, tx))
            Y = np.vstack((Y, ty))
            evaluate_x = np.vstack((evaluate_x, ex))
            evaluate_y = np.vstack((evaluate_y, ey))

        print('start training ... ')
        print('Final X.shape:' + str(X.shape))
        print('Final Y.shape:' + str(Y.shape))
        print('Final evaluate_x.shape:' + str(evaluate_x.shape))
        print('Final evaluate_y.shape:' + str(evaluate_y.shape))
        self.countLabel(Y)

        if config.debugForPrepareData is True:
            for i in range(len(X)):
                # self.drawTrainData(X[i], Y[i])
                pass

        return X, Y, evaluate_x, evaluate_y

    def processData(self, fileName):
        code = fileName.replace('.CSV', '')

        originalDataSaver = OriginalDataSaver(fileName)
        print(code + ' start to originalDataSaver.processFile...')
        originalDataSaver.processFile()

        print(code + ' start to labelProcessor.refreshLabels...')
        labelProcessor = LabelProcessor(code, startDateTime, endDateTime)
        labelProcessor.refreshLabels()

        print(code + ' start to macdProcessor.refreshMACD...')
        macdProcessor = MacdProcessor(code, startDateTime, endDateTime)
        macdProcessor.refreshMACD()
