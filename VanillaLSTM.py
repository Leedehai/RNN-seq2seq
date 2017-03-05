"""
Vanilla LSTM model implemented with TensorFlow
CS224N Project

Author: Haihong Li
Date: March 1, 2017
"""

import tensorflow as tf
import numpy as np
import time
import ipdb
import argparse

import rnn_segment

# The vanilla LSTM model
class VanillaLSTMModel():
    '''
	The classic, standard, vanilla LSTM model for encoder-decoder structure
	i.e. a encoder-decoder RNN using homogeneous LSTM cells.
	'''
    def __init__(self, args, training=True):
		'''
		Initialization function for the class Model.
		Params:
		  args: contains arguments required for the Model creation --
		    args.hidden_size (i.e. the size of hidden state)
		    args.num_layers (default = 1, i.e. no stacking), 
		    args.seq_length, 
		    args.embedding_size, 
		    args.batch_size (i.e. the number of sequences in each batch)
		    args.optimizer_choice (defualt = "rms", also could be "adam", "grad_desc")
		    args.learning_rate, 
		    args.grad_clip
		NOTE Each cell's input is batch_size x 1 x embedding_size
		NOTE Each cell's output is also batch_size x 1 x embedding_size
		'''
		# Store the arguments, and print the important argument values
		self.args = args
		print("VanillaLSTMModel initializer is called..\n" \
		      + "Time: " + time.ctime() + "\n" \
			  + "  args.hidden_size " + str(args.hidden_size) + "\n" \
			  + "  args.num_layers " + str(args.num_layers) + "\n" \
			  + "  args.embedding_size " + str(args.embedding_size) + "\n")
		
		if training:
			print("Traininig..\n")
		else:
			print("This is a session other than training session..\n")
			print("Input batch size = " + str(args.batch_size) + "\n\n")

		# initialize a LSTM cell unit, hidden_size is the dimension of hidden state
		cell = tf.nn.rnn_cell.BasicLSTMCell(args.hidden_size, state_is_tuple=True)

		# TODO: (improve) Multi-layer RNN ocnstruction, if more than one layer
		# cell = rnn_cell.MultiRNNCell([cell] * args.num_layers, state_is_tuple=False)
		
		# TODO: (improve) Dropout layer can be added here
		# Store the recurrent unit
		self.cell = cell

		# TODO: (resolve) do we need to use a fixed seq_length?
		# Input data contains sequences of input tokens of embedding_size dimension
		self.input_data = tf.placeholder(tf.float32, [None, args.seq_length, args.embedding_size])
		# Target data contains sequences of output tokens of embedding_size dimension
		self.target_data = tf.placeholder(tf.float32, [None, args.seq_length, args.embedding_size])

        # Learning rate
		self.lr = tf.Variable(args.learning_rate, trainable=False, name="learning_rate")

		# Initial cell state of LSTM (initialized with zeros)
		# TODO: (improve) might use xavier initializer? There seems to be no a staright-forward way
		self.initial_state = cell.zero_state(batch_size=args.batch_size, dtype=tf.float32)

		# Split inputs according to sequences: a 3D tensor, num_of_seq x seq_length x embedding_size
		# -> list of size seq_length, each of whose element is of num_of_seq x 1 x embedding_size
		if tf.__version__[0:2] == '0.':
			input_data_temp = tf.split(split_dim=1, num_split=args.seq_length, value=self.input_data)
		elif tf.__version__[0:2] == '1.':
			input_data_temp = tf.split(value=self.input_data, num_split=args.seq_length, split_dim=1)
		# list of size seq_length, each of which is num_of_seq x 1 x input_token_size
		# -> list of size seq_length, each of which is num_of_seq x input_token_size
		input_data_list = [tf.squeeze(input=inputs_member, axis=[1]) for inputs_member in input_data_temp]
		del input_data_temp
		
		## This is where the LSTM models differ from each other.
		# call the encoder
		_, self.encoder_final_state = rnn_segment.run(cell=self.cell, 
		                                         inputs=input_data_list, 
												 initial_state=self.initial_state, 
												 feed_previous=False)
		# call the decoder
		self.output_data, _ = rnn_segment.run(cell=self.cell, 
		                                      inputs=input_data_list, 
											  initial_state=self.encoder_final_state, 
											  feed_previous=True)
		
		
		def get_sum_of_cost(output_data, target_data):
			'''
			Calculate the sum of cost of this training batch using cross entropy.
			params:
			output_data: batch_size x seq_length x embedding_size
			target_data: batch_size x seq_length x output_vocab_size (one-hot)
			'''
			# TODO: (unfinished) finsh this function
			return 0


		# Compute the cost scalar: specifically, the average cost per sequence
		sum_of_cost = get_sum_of_cost(output_data=self.output_data, target_data=self.target_data)
		self.cost = tf.div(sum_of_cost, args.batch_size)

		# Get trainable_variables list, and compute the gradients
		# Also clip the gradients if they are larger than args.grad_clip
		trainable_var = tf.trainable_variables()
		print("Number of trainable variables = " + str(len(trainable_var)))
		self.gradients = tf.gradients(self.loss, trainable_variables)
		self.clipped_grads, _ = tf.clip_by_global_norm(self.gradients, args.grad_clip)

		# Using RMSprop, inspired by the LSTM paper of Dr. Alahi, Prof. Saverese, and Prof. Fei-Fei Li
		if args.optimizer_choice == "rms":
			optimizer = tf.train.RMSPropOptimizer(self.lr)
		elif args.optimizer_choice == "adam":
			optimizer = tf.train.AdamOptimizer(self.lr)
		elif args.optimizer_choice == "grad_desc":
			optimizer = tf.train.GradientDescentOptimizer(self.lr)
		else:
			raise ValueError("Optimizer not supported: " + args.optimizer_choice)

		# Train operator
		self.train_op = optimizer.apply_gradients(zip(clipped_grads, trainable_var))

def main():
	'''
	This function is used for unit testing this module.
	    args.hidden_size (i.e. the size of hidden state)
		args.num_layers (default = 1, i.e. no stacking), 
		args.seq_length, 
	    args.embedding_size, 
		args.batch_size (i.e. the number of sequences in each batch)
		args.optimizer_choice (defualt = "rms", also could be "adam", "grad_desc")
		args.learning_rate, 
		args.grad_clip
	'''
	parser = argparse.ArgumentParser()
	# RNN cell hidden state's size
	parser.add_argument('--hidden_size', type=int, default=64,
	                    help='size of RNN cell hidden state')
	# Number of stacked RNN layers. Only a single layer implemented
	parser.add_argument('--num_layers', type=int, default=1,
	                    help='number of stacked RNN layers')
	# Maximum length of each sequence
	parser.add_argument('--seq_length', type=int, default=16,
	                    help='maximum length of each sequence')
	# Embedding size
	parser.add_argument('--embedding_size', type=int, default=32,
	                    help='embedding size of vectors')
	# Batch size
	parser.add_argument('--batch_size', type=int, default=2,
	                    help='number of sequences in a batch')
	# Choice of optimzier
	parser.add_argument('--optimizer_choice', type=str, default='rms',
	                    help='rms (defualt), adam, grad_desc')
	# Learning rate
	parser.add_argument('--learning_rate', type=float, default=0.002,
	                    help='Learning rate')	
	# Gradient clip, i.e. maximum value of gradient amplitute allowed
	parser.add_argument('--grad_clip', type=float, default=10.0,
	                    help='gradient upbound, i.e. maximum value of gradient amplitute allowed')
	# Parse the arguments, and construct the model
	args = parser.parse_args()
	model = VanillaLSTMModel(args)
	# TODO: (improve) maybe we do something more?

if __name__ == "__main__":
	main()