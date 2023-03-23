from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from typing import List, Tuple

from gensim import utils
from gensim.models import KeyedVectors, Word2Vec
from progress.bar import Bar


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
    parser.add_argument("--train", action=BooleanOptionalAction)
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


def main() -> None:
    modelFilePath: str = "models/w2v.gensim"
    args: Namespace = getArgs()

    if args.train:
        train(modelFilePath)
        quit(1)

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

    w2v: Word2Vec = Word2Vec.load(modelFilePath)
    wordVectors: KeyedVectors = w2v.wv

    testWord: str
    with open(file="similarityQueryResults.txt", mode="w") as sqr:
        for testWord in testSimilarity:
            similarWords: List[Tuple[str, float]] = similarityQuery(
                word=testWord, wv=wordVectors
            )
            similarWords: List[str] = [
                ",".join([word, str(similarity)]) + "\n"
                for word, similarity in similarWords
            ]
            similarWords.append("\n")
            sqr.writelines(similarWords)
        sqr.close()


if __name__ == "__main__":
    main()
