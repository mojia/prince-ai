from OriginalDataSaver import OriginalDataSaver
from LabelProcessor import LabelProcessor
from MacdProcessor import MacdProcessor

startDateTime = "2001-10-01 09:35:00"
endDateTime = "2020-10-20 09:35:00"
fileName = 'SH600003.CSV'


def process(csvFileName):
    originalDataSaver = OriginalDataSaver(csvFileName)
    originalDataSaver.processFile()

    code = csvFileName.replace('.CSV', '')

    labelProcessor = LabelProcessor(code, startDateTime, endDateTime)
    labelProcessor.refreshLabels()

    macdProcessor = MacdProcessor(code, startDateTime, endDateTime)
    macdProcessor.refreshMACD()


process(fileName)
