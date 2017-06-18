"""
Model Trainer implemented with TensorFlow 0.12.1, Python 2.7.12
- - - - -
CS224N Project at Stanford Univeristy
Project mentor: Prof. Chirs Manning

Author: Haihong (@Leedehai)
Date: March 7, 2017
- - - - -
Model Trainer instantiates a model, train it and save the resulting parameters, and keep a log.
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

class Trainer():
    '''
    Model trainer.
    '''
    def __init__(self, args, model_choice="V", if_testing=False):
        '''
        Instantiate a model, a save file, and a log text file.
        Params:
            args: contains arguments required for the model creation.
            model_choice: specify the choice of model, default to "VanillaLSTMTransModel".
                V: VanillaLSTMTransModel
                H: HierLSTMTransModel, i.e. Hierarchical LSTM Model
                A/AH: AttenHierLSTMTransModel, i.e. Hierarchical LSTM Model with Attention
        Returns:
            None
        '''
        # Save the args
        self.args = args
        self.if_testing = if_testing

        # Instantiate a model
        build_start_time = time.time()
        if args.continue_training == False:
            # First time training
            vprint(True, "Trainer is called. First time training.")
            vprint(True,"\033[1;m" + "Building computation graph for the model..." + "\033[0;m", color="CYAN")
            if model_choice == "VanillaLSTMTransModel" or model_choice == "V":
                model = VanillaLSTMTransModel(args)
                model_abbr = "V"
            elif model_chice == "AttenVanillaLSTMTransModel" or model_choice == "AV":
                model = AttenVanillaLSTMTransModel(args)
                model_abbr = "AV"
            elif model_choice == "HierLSTMTransModel" or model_choice == "H":
                model = HierLSTMTransModel(args)
                model_abbr = "H"
            elif model_choice + "AttenHierLSTMTransModel" or model_choice == "AH":
                model = AttenHierLSTMTransModel(args)
                model_abbr = "AH"
            else:
                raise ValueError("Model choice: " + str(model_choice) + " is not supported")
            self.model = model
            # Directory to save things
            if args.test == True:
                self.directory = "../RUN_" + model_abbr
                #self.directory = "../TestRUN_" + model_abbr + time.strftime("_%b%d_%H-%M-%S")
            else:
                self.directory = "../RUN_" + model_abbr
            os.mkdir(self.directory)
        else:
            # Continuing training
            if model_choice == "VanillaLSTMTransModel" or model_choice == "V":
                model_abbr = "V"
            elif model_choice == "HierLSTMTransModel" or model_choice == "H":
                model_abbr = "H"
            elif model_choice + "AttenHierLSTMTransModel" or model_choice == "AH" or model_choice == "A":
                model_abbr = "A"
            else:
                raise ValueError("Model choice: " + str(model_choice) + " is not supported")
            self.directory = "../RUN_" + model_abbr
            vprint(True, "Trainer is called. Continuing training.")
            try:
                with open(os.path.join(self.directory, 'args.pkl'), 'r+') as f:
                    saved_args = pickle.load(f)
                    self.args = saved_args
                    # Don't forget this line below
                    self.args.continue_training = True
            except:
                raise ValueError("The specified model is either not trained, damaged,\
                                or in a wrong path. It should be ../RUN_" + model_abbr + "/args.pkl")
            vprint(True, "\033[1;m" + "Rebuilding computation graph for the model..." + "\033[0;m", color="CYAN")
            if model_choice == "VanillaLSTMTransModel" or model_choice == "V":
                model = VanillaLSTMTransModel(saved_args)
            elif model_choice == "HierLSTMTransModel" or model_choice == "H":
                model = HierLSTMTransModel(saved_args)
            elif model_choice + "AttenHierLSTMTransModel" or model_choice == "AH" or model_choice == "A":
                model = AttenHierLSTMTransModel(saved_args)
            self.model = model
        
        build_end_time = time.time()
        vprint(True,"\033[1;m" + "Graph built. Time used: " + str(build_end_time - build_start_time) + " seconds" + "\033[0;m", color="CYAN")

        # Create/open a save file to save things.
        if args.continue_training == False:
            with open(os.path.join(self.directory, 'args.pkl'), 'a') as f:
                pickle.dump(args, f)
                vprint(True, "Arguments saved to file: " + self.directory + "/args.pkl")
        else:
            # Continuing from previous traing, do not write the arguments again.
            pass
        log = open(os.path.join(self.directory, 'log.txt'), 'a') # append from EOF. Create file if not found.
        log.write("Log file: " + self.directory + '\n')
        log.close()
        reduced_log = open(os.path.join(self.directory, 'reduced_log.txt'), 'a') # append from EOF. Create file if not found.
        reduced_log.write("Log file: " + self.directory + '\n')
        
    def train(self, num_epochs=100, save_every_batch=400):
        '''
        Train the model.
        Params:
            num_epochs: number of epochs, defualt to 100
            save_every_batch: period of saving, epoch * data_loader.get_num_batches() + batch_index, defualt to 400.
                NOTE in the current implementation, this argument is unused. I opted to save after each epoch.
        Returns:
            None
        '''
        args = self.args
        decay_rate = 0.95 # You may modify it yourself. decay_rate in (0,1]
        log = open(os.path.join(self.directory, 'log.txt'), 'a')
        reduced_log = open(os.path.join(self.directory, 'reduced_log.txt'), 'a')

        data_loader = Dataloader(batch_size=args.batch_size, 
                                 seq_lengths=[args.input_seq_length, args.target_seq_length], 
                                 token_sizes=[args.input_embedding_size, args.target_token_size],
                                 if_testing=self.if_testing)
        num_batches = data_loader.get_num_batches()
        
        # Tic
        train_start_time = time.time()
        vprint(True, "")
        with tf.Session() as sess:
            if args.continue_training == False:
                # Initialize all varaibles in the computational graph
                # r0.11 or earlier: sess.run(tf.initialize_all_variables())
                sess.run(tf.global_variables_initializer())
                # Add all the variables to the registration list of variables to be saved
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=50)
            else:
                # Access the checkpoint file
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir=self.directory, latest_filename=None)
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=50)
                saver.restore(sess=sess, save_path=ckpt.model_checkpoint_path)
                print ckpt.model_checkpoint_path

            train_loss = 0.0
            # For each epoch
            for e in range(num_epochs):
                # Reset data loader so that it reads from the beginning.
                data_loader.reset()
                print args.continue_training
                vprint(args.continue_training, "\033[1;mContinued training\033[0;m", color="MAG")
                vprint(True, "\033[1;mStepped in epoch e = " + str(e+1) + "\033[0;m", color="MAG")
                # Assign the learning rate (decayed acceleration to the epoch number)
                sess.run(tf.assign(self.model.lr, args.learning_rate * (decay_rate ** e)))
                #Get the initial state of the encoder
                state = sess.run(self.model.initial_state)

                # For each batch in this epoch
                for b in range(num_batches):
                    vprint(True, "Stepped in epoch = " + str(e+1) + ", batch b = " + str(b+1), color="MAG")
                    # Tic
                    batch_start_time = time.time()

                    # Get the input (x) and target (y) data of the current batch
                    vprint(True, "Getting batch.. b = " + str(b+1), color="MAG")
                    # x: input batch. It is a list of length batch_size, each element of which is of size input_seq_length x input_embedding_size
                    # y: target batch. It is a list of length batch_size, each element of which is of size target_seq_length x target_token_size (=1)
                    # yl: target sequences' lengths. It is a list of length batch_size, each element of which is an integer.
                    x, y, yl = data_loader.next_batch()
                    vprint(True, "Got batch. Run the session...", color="MAG")

                    # Feed the input and target data and the initial cell state
                    feed = {self.model.input_data: x, self.model.target_data: y, self.model.target_lens_list: yl, self.model.initial_state: state}

                    # Fetch the loss of the self.model on this batch
                    # output_data is softmaxed. It is a list of length target_seq_length, each element is batch_size x output_vocab_size
                    try:
                        _, train_loss = sess.run([self.model.train_op, self.model.cost], feed_dict=feed)
                        #print output_data[0]
                    except Exception as exception_msg:
                        vprint(True, "sess.run() runtime error.", color="RED")
                        print exception_msg

                    # Toc
                    batch_end_time = time.time()

                    # Print something and write to log
                    log_entry = "epoch {}/{}, global step number {}/{}, \n\
                                 train_loss = {:.5f}, \n\
                                 time/batch = {:.3f} s \n".format(e + 1,
                                                                  num_epochs,
                                                                  e * num_batches + b + 1, 
                                                                  num_epochs * num_batches,
                                                                  train_loss,
                                                                  batch_end_time - batch_start_time)
                    reduced_log_entry = "{} {} {} {} {:.5f}\n".format(e + 1, 
                                                                      num_epochs, 
                                                                      e * num_batches + b + 1, 
                                                                      num_epochs * num_batches, 
                                                                      train_loss)
                    # Print on screen
                    vprint(True, log_entry, color=None)
                    # Append to log.txt and reduced_log.text.
                    log.write(log_entry)
                    reduced_log.write(reduced_log_entry)
                    
                # Save the model after each epoch
                checkpoint_path = os.path.join(self.directory, 'model.ckpt')
                time_stamp_integer = int(time.time())
                saver.save(sess, checkpoint_path, global_step=time_stamp_integer)
                print("Saved to {}".format(checkpoint_path + "-" + str(time_stamp_integer)))
                log.write("Saved to {}".format(checkpoint_path + "-" + str(time_stamp_integer)))
            
            train_end_time = time.time()
            vprint(True, "\033[1;m" + "\nTraining finished. Time used: " + str(train_end_time - train_start_time) + " seconds" + "\033[0;m", color="CYAN")
            log.write("Training finished.\n")
            log.close()
            reduced_log.close()

def main():
	'''
	This function is used for unit testing this module.
	    args.hidden_size (i.e. the size of hidden state)
		args.num_layers (default = 1, i.e. no stacking), 
		args.input_seq_length,
		args.target_seq_length, 
	    args.input_embedding_size, 
		args.output_vocab_size,
		args.batch_size (i.e. the number of sequences in each batch)
		args.optimizer_choice (defualt = "rms", also could be "adam", "grad_desc")
		args.learning_rate, 
		args.grad_clip
		args.test
		args.verbose
	'''
	parser = argparse.ArgumentParser()
	# RNN cell hidden state's size
	parser.add_argument('--hidden_size', type=int, default=96,
	                    help='size of RNN cell hidden state')
	# Number of stacked RNN layers. Only a single layer implemented
	parser.add_argument('--num_layers', type=int, default=1,
	                    help='number of stacked RNN layers')
	# Larger than the max length of each input sequence
	parser.add_argument('--input_seq_length', type=int, default=20,
	                    help='maximum length of each input sequence or larger')
	# Larger than the max of each target sequence
	parser.add_argument('--target_seq_length', type=int, default=20,
	                    help='maximum length of each target sequence or larger')
	# Embedding size of input
	parser.add_argument('--input_embedding_size', type=int, default=96,
	                    help='embedding size of input vectors')
	# Embedding size of output
	parser.add_argument('--output_vocab_size', type=int, default=92,
	                    help='size of output vocabulary')
    # Target token size, should be 1 because it is word index in vocabulary
	parser.add_argument('--target_token_size', type=int, default=1,
	                    help='target token size, normally it should be 1 because it is word index in vocabulary')
	# Batch size
	parser.add_argument('--batch_size', type=int, default=100,
	                    help='number of sequences in a batch')
	# Choice of optimzier
	parser.add_argument('--optimizer_choice', type=str, default='rms',
	                    help='rms (defualt), adam, grad_desc')
	# Learning rate
	parser.add_argument('--learning_rate', type=float, default=0.002,
	                    help='Learning rate')	
	# Gradient clip, i.e. maximum value of gradient amplitute allowed
	parser.add_argument('--grad_clip', type=float, default=None,
	                    help='gradient upbound, i.e. maximum value of gradient amplitute allowed')
	# Model unit testing flag, default to False
	parser.add_argument('-t','--test', action='store_true',
	                    help='only set to true when performing unit test')
	# Verbosity flag, default to False
	parser.add_argument('-v','--verbose', action='store_true',
	                    help='only set to true when you want verbosity')
	# Parse the arguments, and construct the model
	args = parser.parse_args()
	args.test = True # It is unit testing, should be True
	args.verbose = True
	trainer = Trainer(args=args, model_choice='VanillaLSTMTransModel', if_testing=True)
	trainer.train(num_epochs=args.epochs, save_every_batch=50)

if __name__ == "__main__":
    main()
    print("This module: " + __file__ + " is functioning.") # test passed. 01:06 March 9, 2017