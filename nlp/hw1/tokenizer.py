from json import dump
from pathlib import PurePath

from progress.bar import Bar

from utils import loadDataTXT


def createTokenSet(data: list) -> set:
    tokenList: list = []

    with Bar("Generating token list...", max=len(data)) as bar:
        value: str
        t: str
        for value in data:
            tokens: list = value.split(" ")

            for t in tokens:
                tokenList.append(t)

            bar.next()

    return set(tokenList)


def removeIntersection(set1: set, set2: set) -> tuple:
    intersection: set = set1.intersection(set2)

    set1 = set1 - intersection
    set2 = set2 - intersection

    return (set1, set2)


def main():
    positiveDataFile: PurePath = PurePath("positiveSentiment.txt")
    negativeDataFile: PurePath = PurePath("negativeSentiment.txt")

    positiveData: list = loadDataTXT(positiveDataFile)
    negativeData: list = loadDataTXT(negativeDataFile)

    positiveTokens: set = createTokenSet(positiveData)
    negativeTokens: set = createTokenSet(negativeData)

    positiveTokens, negativeTokens = removeIntersection(positiveTokens, negativeTokens)

    positiveTokens: dict = {token: "positive" for token in positiveTokens}
    negativeTokens: dict = {token: "negative" for token in negativeTokens}

    positiveTokens.update(negativeTokens)
    tokens: dict = positiveTokens

    with open("tokens.json", "w") as jsonFile:
        dump(tokens, jsonFile)
        jsonFile.close()


if __name__ == "__main__":
    main()
