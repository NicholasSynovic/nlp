from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from typing import List, Tuple

from gensim import downloader, utils
from gensim.models import KeyedVectors, Word2Vec
from progress.bar import Bar
from scipy.stats import spearmanr
from scipy.stats._stats_py import SignificanceResult


class MyCorpus:
    """An iterator that yields sentences (lists of str).
    Code from https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#sphx-glr-auto-examples-tutorials-run-word2vec-py
    """

    epochCount: int = 1

    def __iter__(self):
        line: str
        corpus_path = "wikitext-103/wiki.train.tokens"
        # https://stackoverflow.com/a/1019572
        num_lines = sum(1 for line in open(corpus_path))

        with Bar(f"Iterating through {corpus_path}...", max=num_lines) as bar:
            for line in open(corpus_path):
                if (line.isspace()) or line[0:1] == "=":
                    bar.next()
                    continue
                line: str = line.strip()
                yield utils.simple_preprocess(line)
                bar.next()

        self.epochCount += 1


def getArgs() -> Namespace:
    parser: ArgumentParser = ArgumentParser(
        prog="COMP 429 NLP HW4",
        usage="Homework 4 solution for COMP 429",
        epilog="Written by Nicholas M. Synovic",
    )
    parser.add_argument(
        "--train",
        action=BooleanOptionalAction,
        help="Initiate training of custom Word2Vec model",
    )
    parser.add_argument(
        "--download-google-news",
        action=BooleanOptionalAction,
        help="Download Google News Word2Vec model and save to disk",
    )

    return parser.parse_args()


def train(modelFilePath: str = "models/w2v.gensim") -> None:
    sentences: MyCorpus = MyCorpus()

    print("Creating Word2Vec model...")
    model: Word2Vec = Word2Vec(sentences=sentences)

    print(f"Saving model to {modelFilePath}...")
    model.save(modelFilePath)


def similarityQuery(
    word: str, wv: KeyedVectors, topN: int = 10
) -> List[Tuple[str, float]]:
    vector = wv[word]
    return wv.most_similar([vector], topn=topN)


def downloadGoogleNews(
    modelFilePath: str = "models/googleNews.keyedvectors.gensim",
) -> None:
    print("Downloading word2vec-google-news-300...")
    model: KeyedVectors = downloader.load(name="word2vec-google-news-300")

    print(f"Saving model to {modelFilePath}...")
    model.save(modelFilePath)


def part1(model: KeyedVectors) -> None:
    outputFilePath: str = "similarityQueryResults.txt"

    testSimilarity: List[str] = [
        "science",
        "math",
        "test",
        "man",
        "woman",
        "king",
        "you",
        "apple",
        "queen",
        "the",
    ]

    print(f"Writing similarity test results to {outputFilePath}...")

    testWord: str
    with open(file=outputFilePath, mode="w") as sqr:
        for testWord in testSimilarity:
            similarWords: List[Tuple[str, float]] = similarityQuery(
                word=testWord, wv=model
            )

            similarWords: List[str] = [
                ",".join([word, str(similarity)]) + "\n"
                for word, similarity in similarWords
            ]

            similarWords.append("\n")
            sqr.writelines(similarWords)
        sqr.close()


def part2(
    model: KeyedVectors, outputFilePath: str = "googleNewsSimilarityResults.txt"
) -> None:
    testSimilarity: List[str] = [
        "human",
        "bird",
        "ball",
        "soccer",
        "tee",
        "tea",
        "England",
        "Trump",
        "tiny",
        "computer",
    ]

    print(f"Writing similarity test results to {outputFilePath}...")

    testWord: str
    with open(file=outputFilePath, mode="w") as sqr:
        for testWord in testSimilarity:
            similarWords: List[Tuple[str, float]] = similarityQuery(
                word=testWord, wv=model, topN=100
            )

            similarWords: List[str] = [
                ",".join([word, str(similarity)]) + "\n"
                for word, similarity in similarWords
            ]

            similarWords.append("\n")
            sqr.writelines(similarWords)
        sqr.close()


def part3(model: KeyedVectors) -> float:
    wordsimData: List[Tuple[str, str, float]] = []
    googleNewsData: List[Tuple[str, str, float]] = []

    wordsimPath: str = "wordsim353_sim_rel/wordsim_similarity_goldstandard.txt"

    print("Reading wordsim file...")
    line: str
    for line in open(file=wordsimPath, mode="r"):
        data: List[str] = line.strip().split(sep="\t")
        foo: Tuple[str, str, float] = (data[0], data[1], float(data[2]))
        wordsimData.append(foo)

    print("Getting similarity of words from the Google News dataset...")
    grouping: Tuple[str, str, float]
    for grouping in wordsimData:
        word1: str = grouping[0]
        word2: str = grouping[1]
        score: float = model.similarity(w1=word1, w2=word2)
        googleNewsData.append((word1, word2, score))

    wordsimScores: List[float] = [score for _, _, score in wordsimData]
    googleNewsScores: List[float] = [score for _, _, score in googleNewsData]

    spearmanScore: SignificanceResult = spearmanr(a=wordsimScores, b=googleNewsScores)

    return spearmanScore.statistic


def part4() -> None:
    pass


def main() -> None:
    customModelFilePath: str = "models/w2v.gensim"
    googleNewsModelFilePath: str = "models/googleNews.keyedvectors.gensim"
    args: Namespace = getArgs()

    if args.train:
        train(customModelFilePath)
        quit(1)

    if args.download_google_news:
        downloadGoogleNews(modelFilePath=googleNewsModelFilePath)
        quit(2)

    customModel: KeyedVectors = Word2Vec.load(customModelFilePath).wv
    googleNewsModel: KeyedVectors = KeyedVectors.load(googleNewsModelFilePath)

    part1(model=customModel)
    part2(model=googleNewsModel)
    print(f"Spearman Score: {part3(model=googleNewsModel)}")


if __name__ == "__main__":
    main()
