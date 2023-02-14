# Homework 2

> Author: Nicholas M. Synovic

## Table of Contents

- [Homework 2](#homework-2)
  - [Table of Contents](#table-of-contents)
  - [About](#about)
  - [Dependencies](#dependencies)
  - [How To Run](#how-to-run)
  - [Methodology](#methodology)
    - [Data Splitting](#data-splitting)
  - [Results](#results)

## About

The homework assignment description can be found in [hw2.pdf](hw2.pdf). The
[`hw2.py`](hw2.py) script is the executable code to run to generate results.

The dataset used for this assignment was downloaded from
[here](https://github.com/dennybritz/cnn-text-classification-tf/tree/master/data/rt-polaritydata).

## Dependencies

To run this code, you will need:

- `Python 3.10`
- `requests`
  - This can be installed by running `pip install -r requirements.txt`

## How To Run

- `python3.10 hw2.py`

## Methodology

### Data Splitting

The dataset was split with the following method:

1. Load in the positive dataset
1. Use the first 70% of indexes to create the positive training dataset
1. Use the next 15% of indexes to create the positive development dataset
1. Use the remaining 15% of index to create the positive testing dataset
1. Repeat steps 1 - 4 and substitute the positive dataset for the negative
   dataset (thus resulting in the same splits percentages for the negative
   training, development, and testing datasets)

The

## Results
