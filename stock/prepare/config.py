
class config:
    backWindowLength = 120
    futureWindow = 240
    enableMACDFeatures = True
    featureDim = backWindowLength * 2 + 1
    debugForPrepareData = False
    debugSampleCount = 50
    epochs = 200
    skipStep = 5
    minSizeSamples = 1000
    kFold = 10

    hidden_layer_1_unit = 500
    activation = 'tanh'
