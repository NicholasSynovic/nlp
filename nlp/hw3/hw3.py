from collections import Counter
from pathlib import PurePath
from typing import List, Tuple

import numpy
from numpy import ndarray
from numpy.random import MT19937, RandomState
from sklearn.model_selection import train_test_split


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


def splitData(
    positiveData: List[str], negativeData: List[str]
) -> List[Tuple[List[str], List[str], List[str]]]:
    positiveTempData = List[str]
    negativeTempData = List[str]

    positiveTempData, positiveTestingData = train_test_split(
        positiveData, test_size=0.15, train_size=0.85, random_state=42, shuffle=True
    )
    positiveTrainingData, positiveDevelopmentData = train_test_split(
        positiveTempData, test_size=0.15, train_size=0.7, random_state=42, shuffle=True
    )
    negativeTempData, negativeTestingData = train_test_split(
        negativeData, test_size=0.15, train_size=0.85, random_state=42, shuffle=True
    )
    negativeTrainingData, negativeDevelopmentData = train_test_split(
        negativeTempData, test_size=0.15, train_size=0.7, random_state=42, shuffle=True
    )

    return [
        (positiveTrainingData, positiveDevelopmentData, positiveTestingData),
        (negativeTrainingData, negativeDevelopmentData, negativeTestingData),
    ]


def createWordList(data: List[str]) -> List[str]:
    wordList: List[str] = []

    document: str
    for document in data:
        wordList += document.split(" ")

    return wordList


def termDocumentFrequency(
    data: List[str], wordSet: set[str], label: int
) -> List[List[int]]:
    tdfDict: dict[str, List[int]] = {document: [label] for document in data}
    tdfList: List[List[str]] = []

    for document in tdfDict:
        c: Counter = Counter(document.split(" "))
        for word in wordSet:
            tdfDict[document].append(c[word])
        tdfList.append(tdfDict[document])

    return tdfList


def main() -> None:
    mt19937: MT19937 = MT19937(42)
    rs: RandomState = RandomState(mt19937)

    positivePath: PurePath = PurePath("positive")
    negativePath: PurePath = PurePath("negative")

    positiveData: List[str] = loadData(filepath=positivePath)
    negativeData: List[str] = loadData(filepath=negativePath)

    data: List[Tuple[List[str], List[str], List[str]]] = splitData(
        positiveData, negativeData
    )

    positiveWordList: List[str] = createWordList(data[0][0])
    negativeWordList: List[str] = createWordList(data[1][0])
    wordList: List[str] = positiveWordList + negativeWordList
    wordSet: set[str] = set(wordList)

    positiveTDF: List[List[int]] = termDocumentFrequency(data[0][0], wordSet, label=1)
    negativeTDF: List[List[int]] = termDocumentFrequency(data[1][0], wordSet, label=0)

    tdf: List[List[int]] = positiveTDF + negativeTDF
    tdfNumpy: ndarray = numpy.array(tdf)
    rs.shuffle(tdfNumpy)

    labels: ndarray = tdfNumpy[:, 0]
    tdfNumpy = tdfNumpy[:, 1:]


if __name__ == "__main__":
    main()
