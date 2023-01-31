from math import floor
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


def tokenize(filepath: PurePath) -> Tuple[List[str], List[str]]:
    tokens: List[str] = []
    with open(file=filepath, mode="r") as sentiment:
        lines: List[str] = sentiment.readlines()
        sentiment.close()

    size: int = len(lines)
    sixtyPercent: float = floor(size * 0.6)

    line: str
    for line in lines[0:sixtyPercent]:
        foo: List[str] = line.split(sep=" ")
        tokens.extend(foo)

    tokens: set[str] = set(tokens)
    tokens: List[str] = list(tokens)

    return (tokens, lines[sixtyPercent::])


def evaluate(
    negativeTokens: List[str],
    positiveTokens: List[str],
    negativeTest: List[str],
    positiveTest: List[str],
) -> Tuple[float, float, float, float]:

    tpLines: List[str] = []
    fpLines: List[str] = []
    tnLines: List[str] = []
    fnLines: List[str] = []

    test: str
    for test in positiveTest:
        splitTest: List[str] = test.split(sep=" ")
        testSet: set[str] = set(splitTest)

        tpTest: List[str] = set(positiveTokens) & testSet
        fpTest: List[str] = set(negativeTokens) & testSet

        if len(tpTest) > len(fpTest):
            tpLines.append(test)
        else:
            fpLines.append(test)

    for test in negativeTest:
        splitTest: List[str] = test.split(sep=" ")
        testSet: set[str] = set(splitTest)

        tnTest: List[str] = set(negativeTokens) & testSet
        fnTest: List[str] = set(positiveTokens) & testSet

        if len(tnTest) > len(fnTest):
            tnLines.append(test)
        else:
            fnLines.append(test)

    truePositive: float = len(tpLines) / len(positiveTest)
    trueNegative: float = len(tnLines) / len(negativeTest)
    falsePositive: float = len(fpLines) / len(positiveTest)
    falseNegative: float = len(fnLines) / len(negativeTest)

    return (truePositive, trueNegative, falsePositive, falseNegative)


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

    positive: Tuple[List[str], List[str]] = tokenize(filepath=positiveSentiment)
    negative: Tuple[List[str], List[str]] = tokenize(filepath=negativeSentiment)

    scores: Tuple[float, float, float, float] = evaluate(
        negativeTokens=negative[0],
        positiveTokens=positive[0],
        negativeTest=negative[1],
        positiveTest=positive[1],
    )

    print(
        f"""
        True Positive Score: {scores[0] * 100}%
        True Negative Score: {scores[1] * 100}%
        False Positive Score: {scores[2] * 100}%
        False Negative Score: {scores[3] * 100}%
    """
    )


if __name__ == "__main__":
    main()
