from Utils import Utils

utils = Utils()


class KPeriod:

    def __init__(self, peak, bottom, kList):
        self.peak = peak
        self.bottom = bottom
        self.kList = kList

    def getPrice(self):
        return utils.meanOfK(self.kList)

    def isPeakPeriod(self):
        return self.peak is not None

    def isBottomPeriod(self):
        return self.bottom is not None

    def getPeakTime(self):
        return self.peak.createTime

    def getBottomTime(self):
        return self.bottom.createTime
