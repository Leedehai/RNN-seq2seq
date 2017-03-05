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
import rnn_segment

# The vanilla LSTM model
class VanillaLSTMModel():
    '''
	The classic, standard, vanilla LSTM model for encoder-decoder structure
	i.e. a encoder-decoder RNN using homogeneous LSTM cells.
	'''
	def __init__(self, args, trainig=True):
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
		      + "Time: " + time.ctime() + "\n\n" \
			  + "  args.hidden_size " + str(args.hidden_size) + "\n" \
			  + "  args.num_layers " + str(args.num_layers) + "\n" \
			  + "  args.embedding_size " + str(args.embedding_size) + "\n")
		
		if training:
			print("Traininig..\n\n")
		else:
			print("This is a session other than training session..\n")
			print("Input batch size = " + str(args.batch_size) + "\n\n")

		# initialize a LSTM cell unit, hidden_size is the dimension of hidden state
		cell = tf.nn.rnn_cell.BasicLSTMCell(args.hidden_size, state_is_tuple=False)

		# TODO: (improve) Multi-layer RNN ocnstruction, if more than one layer
		# cell = rnn_cell.MultiRNNCell([cell] * args.num_layers, state_is_tuple=False)
		
		# TODO: (improve) Dropout layer can be added here
		# Store the recurrent unit
		self.cell = cell

		# TODO: (resolve) do we need to use a fixed seq_length?
		# Input data contains sequences of input tokens of embedding_size dimension (defualt = 1)
		self.input_data = tf.placeholder(tf.float32, [None, args.seq_length, args.embedding_size])
		# Target data contains sequences of output tokens of embedding_size dimension (default = 1)
		self.target_data = tf.placeholder(tf.float32, [None, args.seq_length, args.embedding_size])

        # Learning rate
		self.lr = tf.Variable(args.learning_rate, trainable=False, name="learning_rate")

		# Initial cell state of LSTM (initialized with zeros)
		# TODO: (improve) might use xavier initializer? There seems to be no a staright-forward way
		self.initial_state = cell.zero_state(batch_size=args.batch_size, dtype=tf.float32)

		# call the encoder
		# TODO: (unfinished) should change the form of self.input_data, from a 3D tensor
		#       to a list, with each element of size batch_size x embedding_size
		_, encoder_final_state = rnn_segment_run(cell=self.cell, 
		                                         inputs=self.input_data, 
												 initial_state=self.initial_state, 
												 feed_previous=False)
		# call the decoder
		self.output_data, _ = rnn_segment_run(cell=self.cell, 
		                                      inputs=self.input_data, 
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


		# Compute the cost: specifically, the average cost per sequence
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
