"""
Hierarchical LSTM Sequence-to-Sequence Model implemented with TensorFlow 0.12.1, Python 2.7.12
- - - - -
CS224N Project at Stanford Univeristy
Project mentor: Prof. Chris Manning

Author: Haihong (@Leedehai)
Date: March 5, 2017
- - - - -
Hierarchical LSTM Sequence-to-Sequence Model is a LSTM Sequence-to-Sequence Model that treat the input sequence
into three hierarchies: words (minimallest), sentences, paragraph (the input sequence itself).
NOTE THIS MODULE IS NOT FINISHED AS OF MARCH 15, 2017
"""
# Libraries
import tensorflow as tf
import numpy as np
import time
import argparse

# Custom modules
from RNNChain import RNNChain
from verbose_print import vprint

# Debugger
import pdb

# The Hierarchical LSTM model
class HierLSTMTransModel():
	'''
	Hierarchical LSTM model for encoder-decoder structure, proposed by Jiwei Li, Minh-Thang Luong and Dan Jurafsky at Stanford CS. Assume the hidden state size of word-level and sentence-level are the same, for sake of convenient fine-tuning.
	'''
    def __init__(self, args, training=True):
		'''
		Initialization function for the class Model.
		Params:
		  args: contains arguments required for the Model creation --
		    args.hidden_size (i.e. the size of hidden state)
		    args.num_layers (default = 1, i.e. no stacking), 
		    args.input_seq_length,
			args.target_seq_length, 
		    args.input_embedding_size,
			args.output_vocab_size,
			args.target_token_size (=1, target token is target word's index)
		    args.batch_size (i.e. the number of sequences in each batch),
		    args.optimizer_choice (defualt = "rms", also could be "adam", "grad_desc"),
		    args.learning_rate, 
		    args.grad_clip
			args.test
			args.verbose
		  training: indicates whether this is a training session
		Returns:
		    None
		NOTE Each cell's input is batch_size x 1 x input_embedding_size
		NOTE Each cell's output is batch_size x 1 x hidden_size (needs to be converted)
		'''
		if training == False:
			args.batch_size = 1

		# Store the arguments, and print the important argument values
		self.args = args
		verbose = self.args.verbose
		print("VanillaLSTMTransModel initializer is called..\n" \
		      + "Time: " + time.ctime() + "\n" \
			  + "  args.hidden_size (H) = " + str(self.args.hidden_size) + "\n" \
			  + "  args.input_embedding_size (Di) = " + str(self.args.input_embedding_size) + "\n" \
			  + "  args.output_vocab_size (Vo) = " + str(self.args.output_vocab_size) + "\n" \
			  + "  args.num_layers = " + str(self.args.num_layers) + "\n" \
			  + "  args.optimizer_choice = " + self.args.optimizer_choice + "\n" \
			  + "  args.learning_rate = " + str(self.args.learning_rate) + "\n" \
			  + "  args.grad_clip = " + str(self.args.grad_clip) + "\n")
		
		if training:
			print("This is a training session..")
			print("Input batch size = " + str(self.args.batch_size) + "\n")
		else:
			print("This is a session other than training..")
			print("Input batch size = " + str(self.args.batch_size) + "\n")

		# initialize LSTM cell units, hidden_size is the dimension of hidden state
		word_encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.args.hidden_size, initializer=tf.contrib.layers.xavier_initializer(), sstate_is_tuple=True)
		sent_encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.args.hidden_size, initializer=tf.contrib.layers.xavier_initializer(), sstate_is_tuple=True)
		sent_decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.args.hidden_size, initializer=tf.contrib.layers.xavier_initializer(), sstate_is_tuple=True)
		word_decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.args.hidden_size, initializer=tf.contrib.layers.xavier_initializer(), sstate_is_tuple=True)

		# convert cell's outputs (batch_size x hidden_size for each cell) to batch_size x output_vocab_size
		# y_hat = softmax(tf.add(tf.matmul(cell_output, output_ws), output_bs)), output_bs = zeros, for now
		with tf.variable_scope("vanLSTM_decoder/decoder_accessory"):
			self.output_ws = tf.get_variable("output_ws", [self.args.hidden_size, self.args.output_vocab_size])
			output_affine_map_lambda = lambda cell_output_: tf.matmul(cell_output_, self.output_ws)
			output_converter_lambda = lambda cell_output_: tf.nn.softmax(logits=output_affine_map_lambda(cell_output_), dim=-1) # -1: last
			self.output_affine_map_lambda = output_affine_map_lambda
			self.output_converter_lambda = output_converter_lambda

		# Multi-layer RNN ocnstruction, if more than one layer
		if self.args.num_layers <= 0 or isinstance(self.args.num_layers, int) == False:
			raise ValueError("Specified number of layers is non-positive or is not an integer.")
		elif self.args.num_layers >= 2:
			vprint(True, "Stacked RNN: number of layers = " + str(self.args.num_layers))
			word_encoder_cell = tf.nn.rnn_cell.MultiRNNCell([word_encoder_cell] * self.args.num_layers, state_is_tuple=True)
			sent_encoder_cell = tf.nn.rnn_cell.MultiRNNCell([sent_encoder_cell] * self.args.num_layers, state_is_tuple=True)
			sent_decoder_cell = tf.nn.rnn_cell.MultiRNNCell([sent_decoder_cell] * self.args.num_layers, state_is_tuple=True)
			word_decoder_cell = tf.nn.rnn_cell.MultiRNNCell([word_decoder_cell] * self.args.num_layers, state_is_tuple=True)
		
		# TODO: (improve) Dropout layer can be added here
		# Store the recurrent unit
		self.word_encoder_cell = word_encoder_cell
		self.sent_encoder_cell = sent_encoder_cell
		self.sent_decoder_cell = sent_decoder_cell
		self.word_decoder_cell = word_decoder_cell

		# Create encoder and decoder RNNChain instances
		word_encoder = RNNChain(self.word_encoder_cell, name="hierLSTM_word_encoder", scope="hierLSTM_word_encoder")
		sent_encoder = RNNChain(self.sent_encoder_cell, name="hierLSTM_sent_encoder", scope="hierLSTM_sent_encoder")
		sent_decoder = RNNChain(self.sent_decoder_cell, name="hierLSTM_sent_decoder", scope="heirLSTM_sent_decoder")
		word_decoder = RNNChain(self.word_decoder_cell, name="hierLSTM_word_decoder", scope="hierLSTM_word_decoder")
		self.word_encoder = word_encoder
		self.sent_encoder = sent_encoder
		self.sent_decoder = sent_decoder
		self.word_decoder = word_decoder

		# Input data contains sequences of input tokens of input_embedding_size dimension
		self.input_data = tf.placeholder(tf.float32, [None, self.args.input_seq_length, self.args.input_embedding_size])
		# Target data contains sequences of putput tokens of target_token_size dimension (=1)
		self.target_data = tf.placeholder(tf.int32, [None, self.args.target_seq_length, self.args.target_token_size])
		# Target lengths list contains numbers of non-padding input tokens in each sequence in this batch,
		# each i-th element is a list of integers, indicating the number of non-padding tokens in each sentence, and the list's length indicating the number of non-padding sentences in this i-th sequence (which consists of one or more sentences).
		self.target_lens_list = tf.placeholder(tf.int32, [None, self.args.input_num_sent])

		# Learning rate
		self.lr = tf.Variable(self.args.learning_rate, trainable=False, name="learning_rate")

		# Initial cell state of LSTM (initialized with zeros)
		# TODO: (improve) might use xavier initializer?
		self.initial_word_state = word_encoder_cell.zero_state(batch_size=self.args.batch_size, dtype=tf.float32)
		self.initial_sent_state = sent_encoder_cell.zero_state(batch_size=self.args.batch_size, dtype=tf.float32)

		# Preprocessing the information got from placeholders.
		# First, target_lens_list does not need any further actions.
		target_lens_list = self.target_lens_list
		# Second, input_data and target_data need reshaping.
		# Split inputs and 

    def get_args(self):
        return self.args

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
	#args.test = True
	model = HierLSTMTransModel(args)
	print "arguments:"
	vprint(args.verbose, model.get_args().__dict__, color=None)
	# TODO: (improve) maybe we do something more?

if __name__ == "__main__":
	main()
	print("This module: " + __file__ + " is functioning.")