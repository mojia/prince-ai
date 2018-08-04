from OriginalDataSaver import OriginalDataSaver
from LabelProcessor import LabelProcessor
from MacdProcessor import MacdProcessor

startDateTime = "2001-10-01 09:35:00"
endDateTime = "2020-10-20 09:35:00"
csvFileName = 'SH600015.CSV'


code = csvFileName.replace('.CSV', '')

originalDataSaver = OriginalDataSaver(csvFileName)
print(code + ' start to originalDataSaver.processFile...')
originalDataSaver.processFile()

print(code + ' start to labelProcessor.refreshLabels...')
labelProcessor = LabelProcessor(code, startDateTime, endDateTime)
labelProcessor.refreshLabels()

print(code + ' start to macdProcessor.refreshMACD...')
macdProcessor = MacdProcessor(code, startDateTime, endDateTime)
macdProcessor.refreshMACD()
