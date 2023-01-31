from math import floor
from pathlib import PurePath
from typing import List, Tuple

from numpy import ndarray
from numpy.random import RandomState
from requests import Response, get


def downloadData(url: str, filepath: PurePath) -> None:
    resp: Response = get(url)

    text: str = resp.content.decode(encoding="UTF-8")
    lines: List[str] = text.split(sep="\n")
    lines = [f"{line}\n" for line in lines]

    with open(file=filepath, mode="w") as sentiment:
        sentiment.writelines(lines)
        sentiment.close()


def tokenize(filepath: PurePath) -> Tuple[List[str], List[str]]:
    tokens: List[str] = []
    with open(file=filepath, mode="r") as sentiment:
        lines: List[str] = sentiment.readlines()
        sentiment.close()

    line: str
    for line in lines:
        foo: List[str] = line.split(sep=" ")
        tokens.extend(foo)

    tokens: set[str] = set(tokens)
    tokens: List[str] = list(tokens)

    return tokens


def main() -> None:
    tokens: dict = {}
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

    positiveTokens: List[str] = splitData(filepath=positiveSentiment)
    # negativeTokens: List[str] = tokenize(filepath=negativeSentiment)

    # tokens["positive"] = positiveTokens
    # tokens["negative"] = negativeTokens


if __name__ == "__main__":
    main()
