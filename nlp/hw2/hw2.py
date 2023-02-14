from collections import defaultdict
from json import dumps
from math import floor, log
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

    positiveClassLog = log(len(positiveData) / len(data))
    negativeClassLog = log(len(negativeData) / len(data))

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

    # Upweighting first token
    # for idx in range(len(words)):
    #     if idx == 0:
    #         dataDict[words[idx]] += 2
    #     else:
    #         dataDict[words[idx]] += 1

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
    data: dict[str, List[float, float]] = {word: [0, 0] for word in vocab}

    word: str
    for word in positiveData:
        count: int = positiveData[word] + 1
        positiveWordCount: int = positiveWordFrequency + 1
        data[word][0] = log(count / positiveWordCount)

    for word in negativeData:
        count: int = negativeData[word] + 1
        negativeWordCount: int = negativeWordFrequency + 1
        data[word][1] = log(count / negativeWordCount)

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
                positiveDocumentProbability += 0

            try:
                negativeDocumentProbability += classLikelihoods[word][1]
            except KeyError:
                negativeDocumentProbability += 0

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
    tpDocumentCount: int = 0
    fpDocumentCount: int = 0

    for document in test:
        if test[document][0] == test[document][1]:
            tpDocumentCount += 1
        else:
            fpDocumentCount += 1

    totalDocumentCount: int = tpDocumentCount + fpDocumentCount

    return (
        round((tpDocumentCount / totalDocumentCount) * 100, ndigits=5),
        round((fpDocumentCount / totalDocumentCount) * 100, ndigits=5),
    )


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
    Positive Training Data Size     : {len(positiveTrainingData)} ({round((len(positiveTrainingData)/ len(positiveData)) * 100, ndigits=5)}% of Positive Data)
    Positive Development Data Size  : {len(positiveDevelopmentData)} ({round((len(positiveDevelopmentData)/ len(positiveData)) * 100, ndigits=5)}% of Positive Data)
    Positive Testing Data Size      : {len(positiveTestingData)} ({round((len(positiveTestingData)/ len(positiveData)) * 100, ndigits=5)}% of Positive Data)

    Negative Training Data Size     : {len(negativeTrainingData)} ({round((len(negativeTrainingData)/ len(negativeData)) * 100, ndigits=5)}% of Negative Data)
    Negative Development Data Size  : {len(negativeDevelopmentData)} ({round((len(negativeDevelopmentData)/ len(negativeData)) * 100, ndigits=5)}% of Negative Data)
    Negative Testing Data Size      : {len(negativeTestingData)} ({round((len(negativeTestingData)/ len(negativeData)) * 100, ndigits=5)}% of Negative Data)
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

    positiveTrainingData.extend(positiveDevelopmentData)
    negativeTrainingData.extend(negativeDevelopmentData)

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
