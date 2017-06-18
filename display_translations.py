"""
Result index-to-word displayer implemented with Python 2.7.12
- - - - -
CS224N Project at Stanford Univeristy
Project mentor: Prof. Chris Manning

Author: W.Z.Zhou
Date: March 5, 2017
- - - - -
Read in word indecies and translate them to actual words.
"""
import re
import numpy as np
import random
import os
import pickle
import tensorflow as tf

from verbose_print import vprint

DEFAULT_FILE_PATH = "../data/utils/datasets/new_glove.6B.50d.txt"

#### what we need to change here!!
# DATA_PATH is the raw data we are reading in, i.e, number
# SUM_PATH is the paragraph we are writing into.
DATA_PATH = "../raw_prediction/canonical_seq100_feed_input.txt"
SUM_PATH = "../actual_words/canonical_seq100_feed_input.txt"

def loadDict(filepath=DEFAULT_FILE_PATH, dimensions=50):
    count = 0
    with open(filepath, "r") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            count += 1
        vprint(True, "Vocabulary size = " + str(count), color="BLUE")

    id2tok = {}
    id = 0
    with open(filepath) as ifs:
        for line in ifs:
            line = line.strip()
            if not line:
                continue
            row = line.split()
            token = row[0]
            id2tok[id] = token
            id += 1
    return id2tok

def display():

    id2tok = loadDict(filepath=DEFAULT_FILE_PATH, dimensions=50);

    fout = open(SUM_PATH, "w")
   # print id2tok
    with open(DATA_PATH, "r") as fs:
        for line in fs:
            ind_list = re.findall(r"[\w']+|[():/.,!?;-]", line)
            ind_list = [int(p) for p in ind_list]
            res_sentence = [id2tok[i] for i in ind_list]
            for word in res_sentence:
                fout.write("%s " % word)
            fout.write("\n")
    fout.close()

if __name__ == "__main__":
    display()
