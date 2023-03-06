from collections import Counter
from pathlib import PurePath
from typing import List, Tuple

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


def termDocumentFrequency(data: List[str]) -> None:
    wordList: List[str] = []

    document: str
    for document in data:
        wordList += document.split(" ")

    td: dict[str, List] = {document: [] for document in data}

    wordSet: set[str] = set(wordList)

    for document in td:
        c: Counter = Counter(document.split(" "))
        for word in wordSet:
            td[document].append(c[word])

        print(td[document])


def main() -> None:
    positivePath: PurePath = PurePath("positive")
    negativePath: PurePath = PurePath("negative")

    positiveData: List[str] = loadData(filepath=positivePath)
    negativeData: List[str] = loadData(filepath=negativePath)

    data: List[Tuple[List[str], List[str], List[str]]] = splitData(
        positiveData, negativeData
    )


if __name__ == "__main__":
    main()
