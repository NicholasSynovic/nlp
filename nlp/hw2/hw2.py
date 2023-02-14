from collections import defaultdict
from json import dumps
from math import floor, log10
from pathlib import PurePath
from typing import List, Tuple

from requests import Response, get


def downloadData(url: str, filepath: PurePath) -> None:
    resp: Response = get(url)

    text: str = resp.content.decode(encoding="UTF-8")
    lines: List[str] = text.split(sep="\n")
    lines = [f"{line}\n" for line in lines]

    with open(file=filepath, mode="w") as sentiment:
        sentiment.writelines(lines)
        sentiment.close()


def loadData(filepath: PurePath) -> List[str]:
    """Loads data and removes punctuation"""
    data: List[str]

    with open(filepath, "r") as dataFile:
        data = dataFile.readlines()
        dataFile.close()

    data = [d.strip() for d in data]

    idx: int
    for idx in range(len(data)):
        tokens: List[str] = [
            token.lower() for token in data[idx].split(" ") if token.isalpha()
        ]

        data[idx] = " ".join(tokens)

    return data


def splitData(data: List[str]) -> Tuple[List[str], List[str], List[str]]:
    data: List[str] = list(data)
    dataLength: int = len(data)

    trainingLastIdx: int = floor(dataLength * 0.7)
    validationLastIdx: int = floor(dataLength * 0.85)

    training: List[str] = data[0:trainingLastIdx]
    validation: List[str] = data[trainingLastIdx:validationLastIdx]
    testing: List[str] = data[validationLastIdx:-1]

    return (training, validation, testing)


def computeDocumentFrequency(
    positiveData: List[str], negativeData: List[str]
) -> Tuple[float, float]:
    data: List[str] = positiveData + negativeData

    positiveClassLog = log10(len(positiveData) / len(data))
    negativeClassLog = log10(len(negativeData) / len(data))

    return (positiveClassLog, negativeClassLog)


def createVocabulary(positiveData: List[str], negativeData: List[str]) -> List[str]:
    vocab: List[str] = []
    documents: List[str] = positiveData + negativeData

    document: str
    for document in documents:
        vocab.extend(document.split(" "))

    return vocab


def computeWordFrequency(data: List[str]) -> Tuple[dict[str, int], int]:
    words: List[str] = []
    dataDict: defaultdict[str, int] = defaultdict(int)

    sentence: str
    for sentence in data:
        words.extend(sentence.split(" "))

    word: str
    for word in words:
        dataDict[word] += 1

    dataDict: dict[str, int] = dict(dataDict)

    wordCount: int = 0
    for word in dataDict:
        wordCount += dataDict[word]

    return (dataDict, wordCount)


def computeClassLikelihoods(
    positiveData: dict[str, int],
    negativeData: dict[str, int],
    positiveWordFrequency: int,
    negativeWordFrequency: int,
    vocab: List[str],
) -> dict:
    positiveLikelihood: dict[str, float] = {}
    negativeLikelihood: dict[str, float] = {}

    word: str
    for word in positiveData:
        count: int = positiveData[word] + 1
        positiveWordCount: int = positiveWordFrequency + 1
        positiveLikelihood[word] = positiveLikelihood.get(
            word, log10(count / positiveWordCount)
        )

    for word in negativeData:
        count: int = negativeData[word] + 1
        negativeWordCount: int = negativeWordFrequency + 1
        negativeLikelihood[word] = negativeLikelihood.get(
            word, log10(count / negativeWordCount)
        )

    data = {
        word: (positiveLikelihood.get(word), negativeLikelihood.get(word))
        for word in vocab
        if positiveLikelihood.get(word) != None and negativeLikelihood.get(word) != None
    }
    return data


def trainNaiveBayes(
    positiveTrainingData: List[str], negativeTrainingData: List[str]
) -> Tuple:
    bigVocab: List[str] = createVocabulary(
        positiveData=positiveTrainingData, negativeData=negativeTrainingData
    )

    positiveWordFrequencies, totalPositiveWords = computeWordFrequency(
        data=positiveTrainingData
    )

    negativeWordFrequencies, totalNegativeWords = computeWordFrequency(
        data=negativeTrainingData
    )

    return (
        computeClassLikelihoods(
            positiveData=positiveWordFrequencies,
            negativeData=negativeWordFrequencies,
            positiveWordFrequency=totalPositiveWords,
            negativeWordFrequency=totalNegativeWords,
            vocab=bigVocab,
        ),
        bigVocab,
    )


def testNaiveBayes(
    testingData: List[str],
    testingClass: int,
    classLikelihoods: dict,
    positiveClassLog: float,
    negativeClassLog: float,
) -> dict[str, int, int, float, float]:
    data: dict = {}

    document: str
    for document in testingData:
        positiveDocumentProbability: float = positiveClassLog
        negativeDocumentProbability: float = negativeClassLog
        words: List[str] = document.split(" ")

        word: str
        for word in words:
            try:
                positiveDocumentProbability += classLikelihoods[word][0]
            except KeyError:
                pass

            try:
                negativeDocumentProbability += classLikelihoods[word][1]
            except KeyError:
                pass
            except IndexError:
                pass

        documentClass: int
        if positiveDocumentProbability > negativeDocumentProbability:
            documentClass = 1
        else:
            documentClass = 0

        data[document] = [
            documentClass,
            testingClass,
            positiveDocumentProbability,
            negativeDocumentProbability,
        ]

    return data


def computeAccuracy(test: dict) -> Tuple[float, float]:
    properlyLabelledDocuments: int = 0

    document: str
    for document in test:
        correctLabel: int = test[document][1]
        if test[document][0] == correctLabel:
            properlyLabelledDocuments += 1

    totalDocumentCount: int = len(test.keys())

    hitPercentage: float = properlyLabelledDocuments / totalDocumentCount
    missPercentage: float = 1 - hitPercentage

    hitPercentage *= 100
    missPercentage *= 100

    hitPercentage = round(hitPercentage, ndigits=5)
    missPercentage = round(missPercentage, ndigits=5)

    return (hitPercentage, missPercentage)


def main() -> None:
    positiveTrainingData: List[str]
    positiveDevelopmentData: List[str]
    positiveTestingData: List[str]

    negativeTrainingData: List[str]
    negativeDevelopmentData: List[str]
    negativeTestingData: List[str]

    positiveDocumentLog: float
    negativeDocumentLog: float

    classLikelihoods: dict[str, List[float, float]]
    vocab: List[str]

    positiveSentiment: PurePath = PurePath("positive")
    negativeSentiment: PurePath = PurePath("negative")

    downloadData(
        url="https://raw.githubusercontent.com/dennybritz/cnn-text-classification-tf/master/data/rt-polaritydata/rt-polarity.pos",
        filepath=positiveSentiment,
    )
    downloadData(
        url="https://raw.githubusercontent.com/dennybritz/cnn-text-classification-tf/master/data/rt-polaritydata/rt-polarity.neg",
        filepath=negativeSentiment,
    )

    positiveData: List[str] = loadData(filepath=positiveSentiment)
    negativeData: List[str] = loadData(filepath=negativeSentiment)

    positiveTrainingData, positiveDevelopmentData, positiveTestingData = splitData(
        data=positiveData
    )

    negativeTrainingData, negativeDevelopmentData, negativeTestingData = splitData(
        data=negativeData
    )

    print(
        f"""
    Positive Training Doc. Size     : {len(positiveTrainingData)}   ({round((len(positiveTrainingData)/ len(positiveData)) * 100, ndigits=5)}% of Positive Docs)
    Positive Development Doc. Size  : {len(positiveDevelopmentData)}    ({round((len(positiveDevelopmentData)/ len(positiveData)) * 100, ndigits=5)}% of Positive Docs)
    Positive Testing Doc. Size      : {len(positiveTestingData)}    ({round((len(positiveTestingData)/ len(positiveData)) * 100, ndigits=5)}% of Positive Docs)

    Negative Training Doc. Size     : {len(negativeTrainingData)}   ({round((len(negativeTrainingData)/ len(negativeData)) * 100, ndigits=5)}% of Negative Docs)
    Negative Development Doc. Size  : {len(negativeDevelopmentData)}    ({round((len(negativeDevelopmentData)/ len(negativeData)) * 100, ndigits=5)}% of Negative Docs)
    Negative Testing Doc. Size      : {len(negativeTestingData)}    ({round((len(negativeTestingData)/ len(negativeData)) * 100, ndigits=5)}% of Negative Docs)
    """
    )

    positiveDocumentLog, negativeDocumentLog = computeDocumentFrequency(
        positiveData=positiveTrainingData, negativeData=negativeTrainingData
    )

    classLikelihoods, vocab = trainNaiveBayes(
        positiveTrainingData, negativeTrainingData
    )

    positiveDevelopmentTest: dict = testNaiveBayes(
        testingData=positiveDevelopmentData,
        testingClass=1,
        classLikelihoods=classLikelihoods,
        positiveClassLog=positiveDocumentLog,
        negativeClassLog=negativeDocumentLog,
    )

    with open("positiveDevelopment.json", "w") as jsonFile:
        jsonData: str = dumps(obj=positiveDevelopmentTest, indent=4)
        jsonFile.write(jsonData)
        jsonFile.close()

    negativeDevelopmentTest: dict = testNaiveBayes(
        testingData=negativeDevelopmentData,
        testingClass=0,
        classLikelihoods=classLikelihoods,
        positiveClassLog=positiveDocumentLog,
        negativeClassLog=negativeDocumentLog,
    )

    with open("negativeDevelopment.json", "w") as jsonFile:
        jsonData: str = dumps(obj=negativeDevelopmentTest, indent=4)
        jsonFile.write(jsonData)
        jsonFile.close()

    positiveAccuracy: Tuple = computeAccuracy(test=positiveDevelopmentTest)
    negativeAccuracy: Tuple = computeAccuracy(test=negativeDevelopmentTest)

    print(
        f"""
    True Positive   (dev)   : {positiveAccuracy[0]}%
    False Positive  (dev)   : {positiveAccuracy[1]}%

    True Negative   (dev)   : {negativeAccuracy[0]}%
    False Negative  (dev)   : {negativeAccuracy[1]}%
    """
    )

    positiveTrainingData = positiveTrainingData + positiveDevelopmentData
    negativeTrainingData = negativeTrainingData + negativeDevelopmentData

    positiveDocumentLog, negativeDocumentLog = computeDocumentFrequency(
        positiveData=positiveTrainingData, negativeData=negativeTrainingData
    )

    classLikelihoods, vocab = trainNaiveBayes(
        positiveTrainingData, negativeTrainingData
    )

    positiveTest: dict = testNaiveBayes(
        testingData=positiveTestingData,
        testingClass=1,
        classLikelihoods=classLikelihoods,
        positiveClassLog=positiveDocumentLog,
        negativeClassLog=negativeDocumentLog,
    )

    with open("positiveTest.json", "w") as jsonFile:
        jsonData: str = dumps(obj=positiveTest, indent=4)
        jsonFile.write(jsonData)
        jsonFile.close()

    negativeTest: dict = testNaiveBayes(
        testingData=negativeTestingData,
        testingClass=0,
        classLikelihoods=classLikelihoods,
        positiveClassLog=positiveDocumentLog,
        negativeClassLog=negativeDocumentLog,
    )

    with open("negativeTest.json", "w") as jsonFile:
        jsonData: str = dumps(obj=negativeTest, indent=4)
        jsonFile.write(jsonData)
        jsonFile.close()

    positiveAccuracy: Tuple = computeAccuracy(test=positiveTest)
    negativeAccuracy: Tuple = computeAccuracy(test=negativeTest)

    print(
        f"""
    True Positive   : {positiveAccuracy[0]}%
    False Positive  : {positiveAccuracy[1]}%

    True Negative   : {negativeAccuracy[0]}%
    False Negative  : {negativeAccuracy[1]}%
    """
    )


if __name__ == "__main__":
    main()
