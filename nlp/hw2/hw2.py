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


def loadData(filepath: PurePath, stopwords: PurePath) -> set[str]:
    data: List[str]
    stopwordsData: List[str]

    with open(filepath, "r") as dataFile:
        data = dataFile.readlines()
        dataFile.close()

    with open(stopwords, "r") as stopwordsFile:
        stopwordsData = stopwordsFile.readlines()
        stopwordsFile.close()

    data = [d.strip() for d in data]
    stopwordsData = [d.strip() for d in stopwordsData]

    stopwordsSet: set[str] = set(stopwordsData)

    idx: int
    for idx in range(len(data)):
        tokens: List[str] = set(
            [token for token in data[idx].split(" ") if token.isalpha()]
        )

        tokens = tokens - stopwordsSet

        data[idx] = " ".join(tokens)

    return data


def main() -> None:
    positiveSentiment: PurePath = PurePath("positive")
    negativeSentiment: PurePath = PurePath("negative")
    stopwords: PurePath = PurePath("stopwords")

    downloadData(
        url="https://raw.githubusercontent.com/dennybritz/cnn-text-classification-tf/master/data/rt-polaritydata/rt-polarity.pos",
        filepath=positiveSentiment,
    )
    downloadData(
        url="https://raw.githubusercontent.com/dennybritz/cnn-text-classification-tf/master/data/rt-polaritydata/rt-polarity.neg",
        filepath=negativeSentiment,
    )

    positveData: List[str] = loadData(filepath=positiveSentiment, stopwords=stopwords)
    negativeData: List[str] = loadData(filepath=negativeSentiment, stopwords=stopwords)


if __name__ == "__main__":
    main()
