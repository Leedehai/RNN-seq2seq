"""
Model Tester implemented with TensorFlow 0.12.1, Python 2.7.12
- - - - -
CS224N Project at Stanford Univeristy
Project mentor: Prof. Chris Manning

Author: Haihong (@Leedehai)
Date: March 9, 2017
- - - - -
Model Tester makes predictions based on learned model and calculate losses. You may specify how many sequences you would like to use from the test set.
"""
# Libraries
import tensorflow as tf
import numpy as np
import argparse
import os
import time
import pickle

# Custom modules
from verbose_print import vprint
from Dataloader import Dataloader
from VanillaLSTMTransModel import VanillaLSTMTransModel
from AttenVanillaLSTMTransModel import AttenVanillaLSTMTransModel
#from HierLSTMTransModel import HierLSTMTransModel
#from AttenHierLSTMTransModel import AttenHierLSTMTransModel

# Debugger
import pdb

class Tester():
    '''
    Model tester.
    '''
    def __init__(self, args, model_choice="V", test_batch_size=1):
        '''
        Instantiate a tester.
        Params:
            args: contains arguments required for the model rebuilding.
            model_choice: specify the choice of model, default to "VanillaLSTMTransModel".
                V: VanillaLSTMTransModel
                H: HierLSTMTransModel, i.e. Hierarchical LSTM Model
                A/AH: AttenHierLSTMTransModel, i.e. Hierarchical LSTM Model with Attention
            test_batch_size: the size of batch in testing (not necessarily the same as training batch size), default to 1.
        Returns:
            None.
        '''
        self.args = args
        self.args.model_choice = model_choice
        self.test_batch_size = test_batch_size

        if model_choice == "VanillaLSTMTransModel" or model_choice == "V":
            model_abbr = "V"
        elif model_choice == "AttenVanillaLSTMTransModel" or model_choice == "AV":
            model_abbr = "AV"
        elif model_choice == "HierLSTMTransModel" or model_choice == "H":
            model_abbr = "H"
        elif model_choice == "AttenHierLSTMTransModel" or model_choice == "AH":
            model_abbr = "AH"
        else:
            raise ValueError("Model choice: " + str(model_choice) + " is not supported")
        
        self.directory = "../RUN_" + model_abbr
        try:
            with open(os.path.join(self.directory, 'args.pkl'), 'r+') as f:
                saved_args = pickle.load(f)
                saved_args.batch_size = self.test_batch_size
        except:
            raise ValueError("This model is either not trained, damaged, or in a wrong path, which should be ../RUN_" + model_abbr + "args.pkl")
        
        # Instantiate a model with the saved args
        vprint(True, "\033[1;m" + "Testing. Rebuilding computation graph for the model..." + "\033[0;m", color="CYAN")
        if model_choice == "VanillaLSTMTransModel" or model_choice == "V":
            model = VanillaLSTMTransModel(saved_args, training=False)
        elif model_choice == "AttenVanillaLSTMTransModel" or model_choice == "AV":
            model = AttenVanillaLSTMTransModel(saved_args, training=False)
        elif model_choice == "HierLSTMTransModel" or model_choice == "H":
            model = HierLSTMTransModel(saved_args, training=False)
        elif model_choice + "AttenHierLSTMTransModel" or model_choice == "AH":
            model = AttenHierLSTMTransModel(saved_args, training=False)
        self.model = model
        
        # Instantiate a TensorFlow interactive session
        sess = tf.InteractiveSession()
        
        # Initiate a TensorFlow saver
        saver = tf.train.Saver(tf.global_variables())
        # Get the checkpoint file to load the model
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=self.directory, latest_filename=None)
        # Load the model parameters into session
        vprint(True, "\033[1;m" + "Loading the model parameters..." + "\033[0;m", color="CYAN")
        saver.restore(sess, ckpt.model_checkpoint_path)

        # Link the session to Tester.
        self.sess = sess

    def test(self, printout=False, if_testing=False):
        '''
        Read in a sequence and output a sequence.
        Params:
            printout: if True, print the input, target, and output words.
            if_testing: if True, use random number.
        Returns:
            None
        '''
        # For this testing session
        self.total_loss = 0.0
        self.loss_on_each_seq = []

        rf = open(os.path.join(self.directory, 'test_results.txt'), 'w') # Rewrite
        rf.write("Test log file: " + self.directory + '\n')
        rf.close()

        rtf = open(os.path.join(self.directory, 'test_true_label.txt'), 'w') # Rewrite
        # Acquire the model and session
        model = self.model
        sess = self.sess

        data_loader = Dataloader(batch_size=self.test_batch_size, 
                                 seq_lengths=[self.args.input_seq_length, self.args.target_seq_length],
                                 token_sizes=[self.args.input_embedding_size,
                                 self.args.target_token_size],
                                 usage="test",
                                 if_testing=self.args.test)
        # Reset the pointers in the data loader object
        data_loader.reset()

        #self.num_batches = data_loader.get_num_batches()
        self.num_batches = 1
        rf = open(os.path.join(self.directory, 'test_results.txt'), 'a') # append from EOF. Create file if not found.
        lf = open(os.path.join(self.directory, 'test_log.txt'), 'a') # append from EOF. Create file if not found.
        for b in range(self.num_batches):
            test_start_time = time.time()
            
            # First, Make predictions
            vprint(True, "\nGetting batch.. b = " + str(b+1), color="MAG")
            # x: input batch. It is a list of length test_batch_size, each element of which is of size input_seq_length x input_embedding_size
            # y: target batch. It is a list of length test_batch_size, each element of which is of size target_seq_length x target_token_size (=1)
            # yl: target sequences' lengths. It is a list of length test_batch_size, each element of which is an integer.
            x, y, yl = data_loader.next_batch()
            # Feed into the model and get out the prediction
            vprint(True, "Got batch. Making a batch of prediction...", color="MAG")
            #sess.run(tf.assign(self.model.lr, 0.0))
            feed = {self.model.input_data: x, self.model.target_data: y, self.model.target_lens_list: yl}
            # output_data is softmaxed. It is a list of length target_seq_length, each element is test_batch_size x output_vocab_size
            test_loss, output_data = sess.run([self.model.cost, self.model.output_data], feed_dict=feed)
            
            # print "target batch y:"
            # for k in xrange(self.test_batch_size):
            #     print data_loader.data_file_path
            #     print "-- the sequence # " + str(k+1) + " in this test batch"
            #     print [list(yki) for yki in y[k]]
            
            # For k-th test batch, it is a sequence
            for k in xrange(self.test_batch_size):
                for i in xrange(len(y[k])):
                    yki =  int(y[k][i])
                    rtf.write(str([yki]) + " ")
                rtf.write("\n")
            print("test_loss = " + str(test_loss))
            self.loss_on_each_seq.append(test_loss)
            self.total_loss += test_loss

            # Second, document the predictions
            # For k-th test batch
            for k in xrange(self.test_batch_size):
                # For i-th toekn position
                for i in xrange(len(output_data)):
                    # If too long - I do not care what the output is beyond a certain length limit
                    # Allow the output to exceed a little bit - maybe 1.2 times
                    if i > yl[k] * 1.2:
                        break
                    # word_prob_b is the probability distribution over output_vocab_size. It is of size 1 x output_vocab_size
                    word_prob = output_data[i][k]
                    #print word_prob
                    word_index = tf.argmax(word_prob, axis=0)
                    word_index_singleton = [word_index.eval()]
                    rf.write(str(word_index_singleton) + " ")
                rf.write("\n")
            rf.write("\n")
            test_end_time = time.time()
            vprint(True, "time/batch = {:.3f}".format(test_end_time - test_start_time) + " s", color=None)

            #  NOTE that during testing, batch_size = 1. i.e. test the sequences one by one.
            #  output_data is a list of length target_seq_length, each element of which is of size test_batch_size x output_vocab_size
            #  For each token position i-th
            # for i in xrange(len(output_data)):
            #     # Each output_data_item is NumPy ndarray, of size test_batch_size x output_vocab_size
            #     output_data_item = output_data[i]
            #     # Find the largest probability term as the prediction.
            #     output_data_item_index = tf.argmax(output_data_item, axis=1)
            #     # For each test batch 0-th in this token position
            #     if i >= yl[0]:
            #         break
            #     rf.write(str(output_data_item_index.eval()) + " ")
            # rf.write("\n")
        rf.close()
        rtf.close()
        
        vprint(True, "Total loss of this model is " + str(self.total_loss/self.num_batches))
        lf.write("Total loss: " + str(self.total_loss/self.num_batches) + "\n")
        lf.write("Loss on each sequence: \n")
        for i in xrange(len(self.loss_on_each_seq)):
            lf.write(str(self.loss_on_each_seq[i]) + "\n")
        lf.close()

    def get_total_loss(self):
        '''
        Get total loss in this whole testing session.
        Params:
            (empty)
        Returns:
            total_loss: a scalar
        '''
        return self.total_loss
    
    def get_loss_on_each_seq(self):
        '''
        Get loss on each sequence in this whole testing session.
        Params:
            (empty)
        Returns:
            loss_on_each_seq: a list
        '''
        return self.loss_on_each_seq