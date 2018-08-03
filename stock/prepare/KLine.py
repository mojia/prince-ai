Label_Hold = 3


class KLine:

    def __init__(self,  id, type, code, createTime,
                 openPrice, closePrice, highPrice, lowPrice,
                 volume, turnover):
        self.id = id
        self.type = type
        self.code = code
        self.createTime = createTime
        self.openPrice = openPrice
        self.closePrice = closePrice
        self.highPrice = highPrice
        self.lowPrice = lowPrice

        self.volume = volume
        self.turnover = turnover

        self.label = Label_Hold

    def getPrice(self):
        return (self.openPrice + self.closePrice) * 0.50

    def setLabel(self, label):
        self.label = label

    def setMacdValues(self, dif, dea, bar):
        self.dif = dif
        self.dea = dea
        self.bar = bar

    def __str__(self):
        return str(self.id) + \
            ' ' + str(self.code) + \
            ' ' + str(self.createTime) + \
            ' ' + str(self.openPrice) + \
            ' ' + str(self.closePrice) + \
            ' ' + str(self.label) + '\n'
