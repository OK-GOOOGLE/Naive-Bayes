from math import log
from math import fabs
import pprint # nice dictionaries printing
from collections import Counter
class Training():
    stats = {} # dictionary with stats in following format "{class: {words in documents: their probability}} "
    total = 0 # total number of documents
    uniqueWords = set()
    listOfClasses = {}
    numOfDocuments = 0
'''
input format: 
    class :    value1 , value2, value 3, ......
all values form a document(example).
'''


def parseAndCalcStats_Train(training):
    with open("training.txt", 'r') as file:
        for line in file.readlines():
            training.numOfDocuments+=1
            key, words = line.split('\t')[0], line.split('\t')[1]
            training.stats[key] = training.stats.setdefault(key, {})
            training.listOfClasses[key] = training.listOfClasses.setdefault(key, 0) + 1
            for i in words.split(' '):
                training.uniqueWords.add(i.strip()) # remove whitespaces
    # initialize dictionary with non-zero value to avoid zero-factor problem
    for key in training.stats:
        for word in training.uniqueWords:
            training.stats[key][word.strip()] = 1

    #counting words for every class
    with open("training.txt", 'r') as file:
        for line in file.readlines():
            key, words = line.split('\t')[0], line.split('\t')[1]
            for word in words.split(' '):
                training.stats[key][word.strip()] += 1

    # calculating probability of each word in each class ( P(Xi|C) )
    for key in training.stats:
        for word in training.stats[key]:
            training.stats[key][word] /= len(training.uniqueWords)

    #calculating probability of each class occurrence ( P(C) )
    for key in training.listOfClasses.keys():
        training.listOfClasses[key] /= training.numOfDocuments

    return training.stats, training.listOfClasses

def parseTestData():
    testWordsOccurences = []
    listOfClasses = {}
    with open("test.txt", 'r') as file:
        for line in file.readlines():
            training.numOfDocuments+=1
            key, words = line.split('\t')[0], line.split('\t')[1]
            listOfClasses[key] = listOfClasses.setdefault(key, 0) + 1
            wordsInDocument = {}
            for word in words.split():
                wordsInDocument[word.strip()] = wordsInDocument.setdefault(word.strip(), 0) + 1

            testWordsOccurences.append(wordsInDocument)
    return testWordsOccurences # [{word:occurrences, word:occurrences},{}]


def classifySingleDoc(listOfClasses, elemToClassify, dictStats): # elemToClassify is a dict {word: occurrences}
    classProbability = {}
    for classs in listOfClasses:
        # print(listOfClasses[classs])
        probability = 0 if listOfClasses[classs] == 0 else log(listOfClasses[classs])
        for word in elemToClassify:
            if word in dictStats[classs].keys():
                respectiveAttributeProb = dictStats[classs][word] # P(x_i|C)
                probability += log(respectiveAttributeProb)  # log-sum-exp trick to avoid 0 probability when multiply many small numbers. PLUS sign occurs due to logarithms multiplication rule


        classProbability[classs] = fabs(probability)
    classOfElement = min(classProbability.keys(), key=(lambda key: classProbability[key]))
    return classOfElement


def classifyData(listOfClasses, testData, dictStats): # testData is a list of dictionaries "[{word: occurrences},{..},{..} ...]"
    probabBelongToClass = []
    for document in range(0, len(testData)):
        probabCurrentClass = classifySingleDoc(listOfClasses, testData[document], dictStats)
        probabBelongToClass.append(probabCurrentClass) # filling list with probabilities that current document is of current class

    finalStats = {}
    for i in listOfClasses:
        finalStats[i] = 0

    for i in probabBelongToClass:
        finalStats[i] += 1

    print (finalStats)

training = Training()

#statsClasses -- dict {class : {word: probability within class}}
statsClasses, listOfClasses = parseAndCalcStats_Train(training)
pprint.pprint(listOfClasses)

testData = parseTestData() # list of dictionaries"[{word:occurrences,...},{},{},{}...]"
classifyData(listOfClasses, testData, statsClasses)