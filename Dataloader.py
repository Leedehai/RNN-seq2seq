"""
Data Loader implemented with TensorFlow 0.12.1, Python 2.7.12
- - - - -
CS224N Project at Stanford Univeristy
Project mentor: Prof. Chris Manning

Author: Q.W.Fu, W.Z.Zhou, Haihong (@Leedehai)
Date: March 7, 2017
- - - - -
Responsible for loading data. This module may not be compatible with other datasets.
Module API:
constructor:
  __init__(self, batch_size=10, seq_lengths=[2,3], token_sizes=[20,10], embedding_size=50, if_testing=False)
read the next batch:
  next_batch()
the total number of batches
  num_batches
"""
# Libraries
import numpy as np
import random
import os
import pickle
import tensorflow as tf
import re

DEFAULT_FILE_PATH = "../data/utils/datasets/new_glove.6B.50d.txt" # Stanford GloVe dataset, downsized.
num_entry = 0

# Custom modules
from verbose_print import vprint

# Debugger
#import pdb

def loadEmbeddings(filepath=DEFAULT_FILE_PATH, dimensions=50):
    '''
    Read the embedding mapping.
    '''
    count = 0
    with open(filepath, "r") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            count += 1
        vprint(True, "Vocabulary size = " + str(count), color="BLUE")

    embeddings = np.zeros((count, dimensions))
    tok2id = {}
    id = 0
    with open(filepath) as ifs:
        for line in ifs:
            line = line.strip()
            if not line:
                continue
            row = line.split()
            token = row[0]
            tok2id[token] = id
            data = [float(x) for x in row[1:]]
            if len(data) != dimensions:
                print len(data)
                print dimensions
                raise RuntimeError("wrong number of dimensions")
            embeddings[id] = np.asarray(data)
            id += 1
    # with open("dictionary.txt", "w") as fout:
    # 	for key in tok2id.keys():
    # 		fout.write("%s %s\n" % (key, tok2id[key]))
    return embeddings, tok2id


def read_batch(file, batch_size, complex_maxlen, simple_maxlen, dimensions, embeddings, tok2id):
    global num_entry
    END_TOKEN_EMBEDDING = np.ones(50)
    complex_embeddings = []
    simple_index = []
    #complex_embeddings = np.zeros((batch_size, complex_maxlen, dimensions))
    #simple_embeddings = np.zeros((batch_size, maxlen, dimensions))
    #simple_index = np.zeros((batch_size, simple_maxlen, 1))
    entry_count = 0
    word_count = 0
    space_count = 1
    seen_token = False
    line_count = 0
    inside_embeddings = np.zeros((complex_maxlen, dimensions))
    inside_labels = np.zeros((simple_maxlen, 1))
    current_entry = ""

    labels_mask_list = []

    #complex_index is the a list of len: batch_size, and every element is a nparray of shape (complex_maxlen, 1)
    #inside_complex_labels is every element ot complex_index (list)
    #complex_mask_list is a list of len: bath_size, every element is a number representing the number of effective words in padding
    inside_complex_labels = np.zeros((complex_maxlen, 1))
    complex_index = []
    complex_mask_list = []


    while True:
        line = file.readline()
        #print line
        #line_count += 1
        #print line_count
        line = line.strip()
        line = line.lower()
        line = line.strip('\n')
        #print line
        #print len(line)
        if len(line) > 1 and line[0] is '#' and line[1] is '#' and line[2] is '#'and line[3] is '#'and line[4] is '#':
            #if line[0] is '#' and line[1] is '#' and line[2] is '#':
            seen_token = True
            #print "************************************************** End Token"
            #add padding to complex
            inside_embeddings[word_count] = END_TOKEN_EMBEDDING
            ### added by wanzi
            inside_complex_labels[word_count][0] = tok2id['<Article_End_Token>']
            complex_index.append(inside_complex_labels)
            inside_complex_labels = np.zeros((complex_maxlen, 1))
            complex_mask_list.append(word_count+1)  # word_count + 1 signifies numbers

            complex_embeddings.append(inside_embeddings)
            num_entry += 1
            # print current_entry
            # print num_entry
            if len(complex_embeddings) == batch_size + 1:
                print "########################################################"
                print current_entry
                print num_entry
                print entry_count
                print "########################################################"
            inside_embeddings = np.zeros((complex_maxlen, dimensions))
            word_count = 0
            continue
        if space_count == 2:
            current_entry = line
            entry_count += 1
            space_count += 1
            continue
        if len(line) <= 1 and not seen_token:
            space_count += 1
            #print "------------------------------------------------"
            #print space_count
            continue
        if len(line) <= 1 and seen_token:
            #add padding to simple
            #simple_embeddings[entry_count-1][word_count] = END_TOKEN_EMBEDDING
            inside_labels[word_count][0] = tok2id['<Article_End_Token>']
            simple_index.append(inside_labels)

            ###added by Wanzi
            labels_mask_list.append(word_count+1)

            inside_labels = np.zeros((simple_maxlen, 1))
            word_count = 0
            space_count = 1
            seen_token = False
            if entry_count == batch_size:
                break
        if space_count >= 4 and not seen_token:
            #print "----------------------------------------complex"
            #words = line.split()
            words = re.findall(r"[\w']+|[():/.,!?;-]", line)
            for w in words:
                if word_count == complex_maxlen - 1:
                    break
                if w not in tok2id.keys():
                	inside_complex_labels[word_count][0] = tok2id['<unk>']
                	word_count += 1
                	continue
                inside_complex_labels[word_count][0] = tok2id[w]
                inside_embeddings[word_count] = embeddings[tok2id[w]]
                #print "=======", w, "===== ", word_count
                word_count += 1
        if space_count >= 4 and seen_token:
            #words = line.split()
            #print "----------------------------------------simple"
            words = re.findall(r"[\w']+|[():/.,!?;-]", line)
            #print words
            #print len(words)
            for w in words:
                if word_count == simple_maxlen - 1:
                    break
                if w not in tok2id.keys():
                    inside_labels[word_count][0] = tok2id['<unk>']
                    word_count += 1
                    continue
                inside_labels[word_count][0] = tok2id[w]
                word_count += 1
                #print "=======", w, "===== ", word_count
    #print len(labels_mask_list)
    #print labels_mask_list[0:3]

    return complex_embeddings, complex_index, complex_mask_list

class Dataloader():
    '''
    Dataloader, load data in batch.
    '''
    def __init__(self, batch_size=10, seq_lengths=[2,3], token_sizes=[20,1], usage="train", if_testing=False):
        '''
        Initialization of a data loader.
        Params:
            batch_size: size of batch.
            seq_lengths: a integer list, [input_seq_length, target_seq_length]
            token_sizes: a integer list, [input_embedding_size, target_word_index_size], where target_word_index_size = 1
            usage: what this data loader is used for: "train", "dev", test"
            if_testing: if True, returns random data.
        Returns:
            None
        '''
        #train samples 84973
        #dev samples 10614
        #test samples 10617
        self.batch_size = batch_size
        self.seq_lengths = seq_lengths
        self.complex_length = seq_lengths[0]
        self.simple_length = seq_lengths[1]
        self.token_sizes = token_sizes
        self.embedding_size = token_sizes[0]
        self.usage = usage
        self.if_testing = if_testing
        self.num_batches = int(10614 / batch_size) # Changed from 102696 % batch_size
        #self.if_first_batch = True
        
        vprint(True, "Loading embeddings...", color="BLUE")
        if self.usage == "train":
            #self.data_file = open('../data/train_data.txt')
            self.data_file = open('../data/dev_new_data.txt')
            self.data_file_path = '../data/dev_new_data.txt'
        elif self.usage == "test":
            #self.data_file = open('../data/test_data.txt')
            self.data_file = open('../data/dev_new_data.txt')
            #self.data_file_path = '../data/test_data.txt'
            self.data_file_path = '../data/dev_new_data.txt'
        if self.if_testing == True:
            # Testing code. Returns random numbers. Do not touch..
            self.num_batches = 15
        elif self.if_testing == False:
            # Real work
            self.embeddings, self.tok2id = loadEmbeddings(filepath=DEFAULT_FILE_PATH, dimensions=self.embedding_size)
            vprint(True, "Finished loading embeddings", color="BLUE")

    def reset(self):
        self.data_file.close()
        if self.usage == "train":
            self.data_file = open('../data/dev_new_data.txt')
        elif self.usage == "test":
            #self.data_file = open('../data/test_data.txt')
            self.data_file = open('../data/dev_new_data.txt')
    
    def get_num_batches(self):
        '''
        Get number of batches.
        Params:
            (empty)
        Returns:
            self.num_batches
        '''
        return self.num_batches
    
    def testing_using_artificials(self):
        '''
        For testing only.
        Params:
            (empty)
        Returns:
            x_batch, y_batch, yl_batch: artificial data: input, target, target non-padding lengths.
        '''
        # Testing using artificial numbers
        x_batch = []
        y_batch = []
        yl_batch = []
        # For each sequence in batch
        for i in xrange(self.batch_size):
            # i is the sequence number in this batch
            # test 1: sine-valued targets
            lin = np.linspace(0, 2 * np.pi, (self.seq_lengths)[0])
            x_temp = [np.tile(lin_i, [(self.token_sizes)[0]]) for lin_i in lin]
            x_batch.append(x_temp)
            abs_sinlin = np.abs(100 * np.sin(lin))
            y_temp = [np.tile(int(abs_sinlin_i), [(self.token_sizes)[1]]) for abs_sinlin_i in abs_sinlin]
            if len(y_temp) > (self.seq_lengths)[1]:
                y_temp = y_temp[0:(self.seq_lengths)[1]]
            elif len(y_temp) < (self.seq_lengths)[1]:
                while len(y_temp) < (self.seq_lengths)[1]:
                    y_temp.append(np.dot(0, y_temp[0]))
            y_batch.append(y_temp)
            yl_batch.append((self.seq_lengths)[0] / 3 * 2 - i % 2)
            #yl_batch.append((self.seq_lengths)[0]/3*2 - 2 * np.round((np.random.rand())))
            
            # test 2: constant-valued targets
            # x_batch.append(10 * np.random.rand((self.seq_lengths)[0], (self.token_sizes)[0]))
            # temp_y_batch = np.zeros(((self.seq_lengths)[1], (self.token_sizes)[1]))
            # r = (self.seq_lengths)[1]/2
            # temp_y_batch[0:r] = np.ones((r,1))
            # y_batch.append(temp_y_batch) # Numpy array [[1], ..., [1], [0], ..., [0]]
            # yl_batch.append(5)

        return x_batch, y_batch, yl_batch
        
    def next_batch(self):
        '''
        Get the next one batch from self.data
        Params:
            (empty)
        Returns:
            x_batch: input batch (Python list), batch_size x input_seq_length x input_embedding_size
            y_batch: target batch (Python list), batch_size x target_seq_length x target_word_index_size = 1
        '''
        # Testing code. Returns random numbers. Do not touch..
        if self.if_testing == True:
            x_batch, y_batch, yl_batch = self.testing_using_artificials()
            return x_batch, y_batch, yl_batch
            
        # For real work
        x_batch, y_batch, y_mask_list_batch = read_batch(file=self.data_file, batch_size=self.batch_size, complex_maxlen=self.complex_length,simple_maxlen=self.simple_length, 
                                      dimensions=self.embedding_size, embeddings=self.embeddings, tok2id=self.tok2id)
        
        return x_batch, y_batch, y_mask_list_batch


BATCH_SIZE = 50
if __name__ == "__main__":
    # Unit testing using real data.
    # Needs a proper datafile.
    data_loader = Dataloader(batch_size=50, seq_lengths=[10,10],token_sizes=[50,1], usage = "test")

    for i in xrange(60):
        print i
        x_batch, y_batch, y_mask_list_batch = data_loader.next_batch()
        print "len of mask_list: ", len(y_mask_list_batch)
        print y_mask_list_batch[0:5]
        print y_batch[0][0:10,0]
        print y_batch[1][0:10,0]
        #print x_batch[0][10:20,0]

        if len(x_batch) > BATCH_SIZE:
            print num_entry
    # print x_batch1[0][10:20,0]
    # print x_batch1[1][10:20,0]
    # print x_batch2[0][10:20,0]
    # print x_batch2[1][10:20,0]
    # print y_batch1[0][10:20,0]
    # print y_batch1[1][10:20,0]
    # print y_batch2[0][10:20,0]
    # print y_batch2[1][10:20,0]
    print("The framework of this module: " + __file__ + " functioning. Not guaranteed.") # test passed. 20:50 March 9, 2017
