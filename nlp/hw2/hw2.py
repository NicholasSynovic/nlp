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


def loadData(filepath: PurePath, stopWords: PurePath) -> set[str]:
    """Loads data and removes stop words + punctuation"""
    data: List[str]
    stopWordsData: List[str]

    with open(filepath, "r") as dataFile:
        data = dataFile.readlines()
        dataFile.close()

    with open(stopWords, "r") as stopWordsFile:
        stopWordsData = stopWordsFile.readlines()
        stopWordsFile.close()

    data = [d.strip() for d in data]
    stopWordsData = [d.strip() for d in stopWordsData]

    stopWordsSet: set[str] = set(stopWordsData)

    idx: int
    for idx in range(len(data)):
        tokens: List[str] = set(
            [token for token in data[idx].split(" ") if token.isalpha()]
        )

        tokens = tokens - stopWordsSet

        data[idx] = " ".join(tokens)

    return set(data)


def splitData(data: set[str]) -> Tuple[List[str], List[str], List[str]]:
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

    positiveClassLog = abs(log10(len(positiveData) / len(data)))
    negativeClassLog = abs(log10(len(negativeData) / len(data)))

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
    data: dict[str, List[float, float]] = {word: [1, 1] for word in vocab}

    word: str
    for word in positiveData:
        count: int = positiveData[word] + 1
        positiveWordCount: int = positiveWordFrequency + 1
        data[word][0] = abs(log10(count / positiveWordCount))

    for word in negativeData:
        count: int = negativeData[word] + 1
        negativeWordCount: int = negativeWordFrequency + 1
        data[word][1] = abs(log10(count / negativeWordCount))

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
    stopWords: PurePath = PurePath("stopWords")

    downloadData(
        url="https://raw.githubusercontent.com/dennybritz/cnn-text-classification-tf/master/data/rt-polaritydata/rt-polarity.pos",
        filepath=positiveSentiment,
    )
    downloadData(
        url="https://raw.githubusercontent.com/dennybritz/cnn-text-classification-tf/master/data/rt-polaritydata/rt-polarity.neg",
        filepath=negativeSentiment,
    )

    positiveData: set[str] = loadData(filepath=positiveSentiment, stopWords=stopWords)
    negativeData: set[str] = loadData(filepath=negativeSentiment, stopWords=stopWords)

    positiveTrainingData, positiveDevelopmentData, positiveTestingData = splitData(
        data=positiveData
    )

    negativeTrainingData, negativeDevelopmentData, negativeTestingData = splitData(
        data=negativeData
    )

    print(
        f"""
    Positive Training Data Size: {len(positiveTrainingData)} ({round((len(positiveTrainingData)/ len(positiveData)) * 100, ndigits=5)}% of Positive Data)
    Positive Development Data Size: {len(positiveDevelopmentData)} ({round((len(positiveDevelopmentData)/ len(positiveData)) * 100, ndigits=5)}% of Positive Data)
    Positive Testing Data Size: {len(positiveTestingData)} ({round((len(positiveTestingData)/ len(positiveData)) * 100, ndigits=5)}% of Positive Data)

    Negative Training Data Size: {len(negativeTrainingData)} ({round((len(negativeTrainingData)/ len(negativeData)) * 100, ndigits=5)}% of Negative Data)
    Negative Development Data Size: {len(negativeDevelopmentData)} ({round((len(negativeDevelopmentData)/ len(negativeData)) * 100, ndigits=5)}% of Negative Data)
    Negative Testing Data Size: {len(negativeTestingData)} ({round((len(negativeTestingData)/ len(negativeData)) * 100, ndigits=5)}% of Negative Data)
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


if __name__ == "__main__":
    main()
