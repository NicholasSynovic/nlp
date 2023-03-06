from pathlib import PurePath
from typing import List

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


def main() -> None:
    tempData: List[str]
    testingData: List[str]
    trainingData: List[str]
    developmentData: List[str]

    positivePath: PurePath = PurePath("positive")
    negativePath: PurePath = PurePath("negative")

    positiveData: List[str] = loadData(filepath=positivePath)
    negativeData: List[str] = loadData(filepath=negativePath)
    data: List[str] = positiveData + negativeData

    tempData, testingData = train_test_split(
        data, test_size=0.15, train_size=0.85, random_state=42, shuffle=True
    )
    trainingData, developmentdata = train_test_split(
        tempData, test_size=0.15, train_size=0.70, random_state=42, shuffle=True
    )


if __name__ == "__main__":
    main()
