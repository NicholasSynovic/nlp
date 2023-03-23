from gensim import utils
from gensim.models import Word2Vec
from progress.bar import Bar


class MyCorpus:
    """An iterator that yields sentences (lists of str).
    Code from https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#sphx-glr-auto-examples-tutorials-run-word2vec-py
    """

    epochCount: int = 1

    def __iter__(self):
        corpus_path = "wikitext-103/wiki.train.tokens"
        num_lines = sum(1 for line in open(corpus_path))

        with Bar(f"Iterating through {corpus_path}...", max=num_lines) as bar:
            line: str
            for line in open(corpus_path):
                if (line.isspace()) or line[0:1] == "=":
                    bar.next()
                    continue
                line: str = line.strip()
                yield utils.simple_preprocess(line)
                bar.next()

        self.epochCount += 1


def train(modelFilePath: str = "model/w2v.gensim") -> None:
    sentences: MyCorpus = MyCorpus()

    print("Creating Word2Vec model...")
    model: Word2Vec = Word2Vec(sentences=sentences)

    print(f"Saving model to {modelFilePath}...")
    model.save(modelFilePath)


def main() -> None:
    pass


if __name__ == "__main__":
    main()
