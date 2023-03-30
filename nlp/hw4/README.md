# Homework 4

> Author: Nicholas M. Synovic

## Table of Contents

- [Homework 4](#homework-4)
  - [Table of Contents](#table-of-contents)
  - [About](#about)
    - [Datasets](#datasets)
  - [Dependencies](#dependencies)
  - [How To Run](#how-to-run)
    - [Training Custom Model](#training-custom-model)
    - [Downloading the Google News Word2Vec model](#downloading-the-google-news-word2vec-model)
    - [Running the Homework Assignment](#running-the-homework-assignment)
    - [Running Tests](#running-tests)
  - [Part 1 Results](#part-1-results)
    - [Similarity Scores](#similarity-scores)
  - [Part 2 Results](#part-2-results)
  - [Part 3 Results](#part-3-results)
  - [Part 4 Results](#part-4-results)

## About

The homework assignment description can be found in [hw4.pdf](hw4.pdf). The
[`hw4.py`](hw4.py) script is the executable code to run to generate results.

### Datasets

I used the
[WikiText 103 dataset](https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/)
to train my custom model and the
[WordSim353 dataset](http://alfonseca.org/eng/research/wordsim353.html) to test
simularity of the Google News Word2VecModel offered by `gensim`.

Both of these datasets can be downloaded by running the `downloadCorpus.bash`
script.

## Dependencies

To run this code, you will need:

- `Python 3.10`
- `gensim`
- `progress`
- `scipy`

These can be installed by running `pip install -r requirements.txt`

## How To Run

This program is dependent upon you having run `./downloadCorpus.bash` prior to
execution.

### Training Custom Model

- `python3.10 hw4.py --train`

**NOTE**: Training this model takes a long time. I left in a progress bar to
help track the training progress. The model is saved within the `models`
directory.

### Downloading the Google News Word2Vec model

- `python3.10 hw4.py --download-google-news`

**NOTE**: The model is saved within the `models` directory.

### Running the Homework Assignment

- `python3.10 hw4.py`

### Running Tests

- `python3.10 hw4.py`

**NOTE**: Three files are generated from this program:

- `analogiesResult.txt` (for part 4 of the homework assignment)
- `googleNewsSimilarityResults.txt` (for part 2 of the homework assignment)
- `similarityQueryResults.txt` (for part 1 of the homework assignment)

## Part 1 Results

I generated the similarity scores for the following words using my custom
Word2Vec vectorizer trained on the
[WikiText 103 dataset](https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/):

- `science, math, test, man, woman, king, you, apple, queen, the`

### Similarity Scores

```text
science,1.0
psychology,0.7265394330024719
sociology,0.6986805200576782
economics,0.6870989203453064
physics,0.6609295010566711
anthropology,0.6607807874679565
sociological,0.6577431559562683
humanities,0.6570234298706055
sciences,0.6561775207519531
philosophy,0.6493229269981384

math,1.0
mathematics,0.8195533752441406
maths,0.7746661305427551
linguistics,0.7302621006965637
humanities,0.7247207760810852
psychology,0.7245760560035706
curricula,0.7187387943267822
majoring,0.7025998830795288
vocational,0.702498733997345
textbook,0.701054573059082

test,1.0000001192092896
tests,0.8081120848655701
odi,0.646497905254364
wicket,0.6329720616340637
innings,0.6083503365516663
headingley,0.6073567271232605
lindwall,0.6055846214294434
tallon,0.601289689540863
bradman,0.6006634831428528
testing,0.5965350270271301

man,0.9999998807907104
woman,0.7337713241577148
girl,0.6431003212928772
person,0.6354420185089111
boy,0.6331692337989807
lad,0.6003423929214478
someone,0.5933479070663452
soldier,0.5708504915237427
dog,0.5534687638282776
creature,0.551029622554779

woman,1.0000001192092896
girl,0.7855058908462524
man,0.7337712049484253
prostitute,0.7230261564254761
herself,0.6885742545127869
child,0.683616042137146
pimp,0.6628584265708923
lover,0.6582010984420776
maid,0.6567032933235168
person,0.6434125304222107

king,1.0000001192092896
prince,0.7522253394126892
regent,0.7270032167434692
emperor,0.7183892726898193
mercians,0.7179754376411438
monarch,0.7122443914413452
pretender,0.7061838507652283
sigismund,0.696340799331665
haakon,0.6922058463096619
murchada,0.6905379891395569

you,1.0
we,0.8847972750663757
me,0.8028920888900757
myself,0.8016808032989502
somebody,0.7299163937568665
my,0.6794831156730652
yourself,0.6793534159660339
everybody,0.669592022895813
your,0.6659077405929565
really,0.6550812125205994

apple,1.0000001192092896
smartphone,0.6615980267524719
macintosh,0.6389804482460022
app,0.6357181072235107
walmart,0.6287581920623779
microsoft,0.6265473961830139
blackberry,0.6206438541412354
intel,0.6176124811172485
samsung,0.6169895529747009
iphone,0.6153284311294556

queen,1.0
princess,0.7680444121360779
empress,0.7019802927970886
bedchamber,0.6822462677955627
consort,0.6804032921791077
king,0.6695728302001953
dowager,0.6657602787017822
duchess,0.6556864380836487
victoria,0.6490769386291504
coburg,0.6477525234222412

the,1.0000001192092896
this,0.6752980947494507
its,0.673119068145752
their,0.5255526304244995
another,0.5240440368652344
which,0.517024576663971
itself,0.5066283941268921
of,0.5031510591506958
each,0.4979490041732788
full,0.48043012619018555
```

## Part 2 Results

I tested the following words for the top 100 similar words against the Google
News Word2Vec model from `gensim`:

`human, bird, ball, soccer, tee, tea, England, Trump, tiny, computer`

The results of these words can be found in
[`googleNewsSimilarityResults.txt`](googleNewsSimilarityResults.txt)

The Google News Word2Vec model considers these words similar for two reasons:

1. When learning from the Google News dataset, the Word2Vec model can only learn
   so much about the relationships between words. In other words, if a
   relationship between two words does not exist or is uncommon within the
   dataset, the Word2Vec model will not identify that relationship.
1. The cosine distance between the word embeddings (when treated as vectors) are
   most similar compared to other words. Cosine distance is a measurement of how
   parallel two vectors are, regardless of magnitude. Thus, the more similar two
   words are, the more parallel their vectors are in a multi-dimensional space.

## Part 3 Results

The Spearman rank correlation coefficient measured by comparing the similarity
between word pairs from the
[WordSim353 dataset](http://alfonseca.org/eng/research/wordsim353.html) on the
Google News Word2Vec model was: **0.7717239276951675**

## Part 4 Results

I propose the following analogies, their estimated similarity results, and the
actual similarity results:

| **Analogy**          | **Estimated Result** | **Actual Result** |
| -------------------- | -------------------- | ----------------- |
| king - man + woman   | queen                | king              |
| king - country       | man                  | king              |
| sky - sun + moon     | night                | moon              |
| bicycle + engine     | motorcycle           | engine            |
| dog + whiskers - fun | cat                  | whiskers          |

<!-- Table generated with https://www.tablesgenerator.com/markdown_tables-->
