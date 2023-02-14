from math import floor
from pathlib import PurePath
from typing import List, Tuple
from collections import defaultdict

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


def computeWordFrequency(data: List[str])   -> Tuple[dict[str, int], int]:
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



def main() -> None:
    positiveTrainingFrequency: dict[str, int]
    negativeTrainingFrequency: dict[str, int]
    positiveTrainingWordCount: int
    negativeTrainingWordCount: int

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

    positveData: set[str] = loadData(filepath=positiveSentiment, stopWords=stopWords)
    negativeData: set[str] = loadData(filepath=negativeSentiment, stopWords=stopWords)

    positiveSplits: Tuple[set[str], set[str], set[str]] = splitData(data=positveData)
    negativeSplits: Tuple[set[str], set[str], set[str]] = splitData(data=negativeData)

    positiveTrainingFrequency, positiveTrainingWordCount = computeWordFrequency(data=positiveSplits[0])

    negativeTrainingFrequency, negativeTrainingWordCount = computeWordFrequency(data=negativeSplits[0])


if __name__ == "__main__":
    main()
