from pathlib import PurePath
from typing import List

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
    """Loads data and removes stop words"""
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

    return data


def main() -> None:
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

    positveData: List[str] = loadData(filepath=positiveSentiment, stopWords=stopWords)
    negativeData: List[str] = loadData(filepath=negativeSentiment, stopWords=stopWords)


if __name__ == "__main__":
    main()
