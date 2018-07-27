from math import log
import operator
import pickle


def calcEnt(dataset):
    numSamples = len(dataset)
    labelCounts = {}
    for record in dataset:
        label = record[-1]
        if label not in labelCounts.keys():
            labelCounts[label] = 0
        labelCounts[label] += 1

    entropy = 0.0
    for key in labelCounts:
        probability = float(labelCounts[key]) / numSamples
        entropy -= probability * log(probability, 2)
    # print('labelCounts:\n' + str(labelCounts))
    return entropy


def getSubSetOfExcludingFeature(dataset, featureIndex, featureValue):
    subSet = []
    for row in dataset:
        if row[featureIndex] == featureValue:
            subRow = row[:featureIndex]
            subRow.extend(row[featureIndex + 1:])
            subSet.append(subRow)

    return subSet


def chooseBestFeature(dataset):
    numFeatures = len(dataset[0]) - 1
    entD = calcEnt(dataset)
    bestInfoGain = 0.0
    bestFeature = -1

    for i in range(numFeatures):
        feature_i_values = [example[i] for example in dataset]
        print('feature_i_values\n' + str(feature_i_values))

        values = set(feature_i_values)
        feature_i_ent = 0.0
        for value in values:
            subSet = getSubSetOfExcludingFeature(dataset, i, value)
            prob_i = len(subSet) / float(len(dataset))
            feature_i_ent += prob_i * calcEnt(subSet)
        infoGain_i = entD - feature_i_ent
        if infoGain_i > bestInfoGain:
            bestInfoGain = infoGain_i
            bestFeature = i

    return bestFeature


def createTree(dataset, labels):
    classList = [example[-1] for example in dataset]
    if classList.count(classList[0]) == len(classList)
        return classList[0]

    if len(dataset[0]) == 1:
        return mostClass(classList)

    bestFeature = chooseBestFeature(dataset)
    bestFeatureLabel = labels[bestFeature]

    tree = {bestFeatureLabel: {}}

    del (labels[bestFeature])
    bestFeatureValues = [example[bestFeature] for example in dataset]
    featureValuesSet = set(bestFeatureValues)

    for value in featureValuesSet:
        subSet = getSubSetOfExcludingFeature(dataset, bestFeature, value)
        subLabels = labels[:]

        tree[bestFeatureLabel][value] = createTree(subSet, subLabels)
    return tree


def classify(inputTree, featureLabels, testFeatures):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featureIndex = featureLabels.index(firstStr)

    for firstStr_value in secondDict.keys():
        if testFeatures[featureIndex] == firstStr_value:
            if type(secondDict[firstStr_value]).__name__ == 'dict':
                classLabel = classify(
                    secondDict[firstStr_value], featureLabels, testFeatures)
            else:
                classLabel = secondDict(firstStr_value)

    return classLabel


def mostClass(classList):
    classCount = {}
    for class_i in classList:
        if class_i not in classCount.keys():
            classCount[class_i] = 0
        classCount[class_i] += 1
    print('classCount\n' + str(classCount))

    sortedClassCount = sorted(classCount.iteriitems(),
                              key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]


def createDataset():
    dataset = [
        # age [Young, Mid, Old]
        # sex [Male,Female]
        # have child [Yes,No]
        # live [Survived, died]
        ['Y', 'M', 'N', 'S'],
        ['Y', 'M', 'Y', 'D'],
        ['Y', 'M', 'N', 'S'],
        ['Y', 'M', 'Y', 'D'],
        ['Y', 'F', 'N', 'S'],
        ['Y', 'F', 'N', 'S'],
        ['Y', 'F', 'N', 'S'],
        ['Y', 'F', 'N', 'S'],
        ['M', 'M', 'Y', 'D'],
        ['M', 'M', 'N', 'D'],
        ['M', 'M', 'Y', 'D'],
        ['M', 'M', 'N', 'D'],
        ['M', 'F', 'Y', 'S'],
        ['M', 'F', 'Y', 'S'],
        ['M', 'F', 'N', 'S'],
        ['M', 'F', 'Y', 'S'],
        ['O', 'M', 'Y', 'D'],
        ['O', 'M', 'N', 'D'],
        ['O', 'M', 'N', 'D'],
        ['O', 'M', 'Y', 'D'],
        ['O', 'F', 'N', 'S'],
        ['O', 'F', 'N', 'S'],
        ['O', 'F', 'Y', 'S'],
        ['O', 'F', 'Y', 'S']
    ]
    return dataset


# print(calcEnt(D))
# print(getSubSetOfExcludingFeature(D, 1, 'F'))
# print(chooseBestFeature(D))

if __name__ == '__main__':
    D = createDataset()
    labels = [example[-1] for example in dataset]
    createTree()
