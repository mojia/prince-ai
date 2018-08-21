
class Config:

    def __init__(self):
        self.debugForPrepareData = False

        self.backWindowLength = 120 if self.debugForPrepareData is False else 120

        self.epochs = 1000
        self.skipStep = 5 if self.debugForPrepareData is False else 10
        self.minSizeSamples = 1000
        self.kFold = 10

        self.hidden_layer_1_unit = 400
        # sigmod stuck; selu is ok;
        self.activation = 'relu'

        self.addBarFeatures = False

        self.futureWindow = 240
        self.debugSampleCount = 12

    def getDataDim(self):
        count = self.backWindowLength
        if self.addBarFeatures is True:
            count += self.backWindowLength
        else:
            pass

        count += 1
        return count
