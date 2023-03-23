import gensim.models
from gensim import utils


class MyCorpus:
    """An iterator that yields sentences (lists of str).
    Code from https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#sphx-glr-auto-examples-tutorials-run-word2vec-py
    """

    def __iter__(self):
        corpus_path = "wikitext-103/wiki.train.tokens"
        line: str
        for line in open(corpus_path):
            if (line.isspace()) or line[0:1] == "=":
                continue
            line: str = line.strip()
            yield utils.simple_preprocess(line)


sentences = MyCorpus()
model = gensim.models.Word2Vec(sentences=sentences)
