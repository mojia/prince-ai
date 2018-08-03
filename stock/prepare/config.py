
class config:
    trainSampleCount = 100000
    evaluateCount = 40000
    backWindowLength = 60
    futureWindow = 240
    enableMACDFeatures = True
    featureDim = backWindowLength * 2
    debugForPrepareData = False
    epochs = 800
    skipStep = 12
