
class Utils:

    def __init__(self):
        pass

    def meanOfK(self, listOfK):
        sum = 0
        for k in listOfK:
            closePrice = k.closePrice
            sum += float(closePrice)

        return sum * 1.0 / len(listOfK)
