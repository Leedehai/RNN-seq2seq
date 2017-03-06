"""
Vanilla LSTM model implemented with TensorFlow 0.12.1, Python 2.7.12
- - - - -
CS224N Project at Stanford Univeristy
Project mentor: Prof. Chirs Manning

Author: Haihong Li
Date: March 1, 2017
"""

import tensorflow as tf
import numpy as np
import time
import argparse

import pdb

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
		    args.input_embedding_size
			args.output_vocab_size
		    args.batch_size (i.e. the number of sequences in each batch)
		    args.optimizer_choice (defualt = "rms", also could be "adam", "grad_desc")
		    args.learning_rate, 
		    args.grad_clip
		NOTE Each cell's input is batch_size x 1 x input_embedding_size
		NOTE Each cell's output is batch_size x 1 x hidden_size (needs to be converted)
		'''
		# Store the arguments, and print the important argument values
		self.args = args
		print("VanillaLSTMModel initializer is called..\n" \
		      + "Time: " + time.ctime() + "\n" \
			  + "  args.hidden_size (H) = " + str(args.hidden_size) + "\n" \
			  + "  args.input_embedding_size (Di) = " + str(args.input_embedding_size) + "\n" \
			  + "  args.output_vocab_size (Do) = " + str(args.output_vocab_size) + "\n" \
			  + "  args.num_layers = " + str(args.num_layers) + "\n" \
			  + "  args.optimizer_choice = " + args.optimizer_choice + "\n" \
			  + "  args.learning_rate = " + str(args.learning_rate) + "\n" \
			  + "  args.grad_clip = " + str(args.grad_clip) + "\n")
		
		if training:
			print("This is a training session..")
			print("Input batch size = " + str(args.batch_size) + "\n\n")
		else:
			print("This is a session other than training..")
			print("Input batch size = " + str(args.batch_size) + "\n\n")

		# initialize a LSTM cell unit, hidden_size is the dimension of hidden state
		# TODO: (resolve) are the two kinds of cell's hidden state of the same size?
		encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(args.hidden_size, state_is_tuple=True)
		decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(args.hidden_size, state_is_tuple=True)
		
		# convert cell's outputs (batch_size x hidden_size for each cell) to batch_size x output_vocab_size
		# y_hat = softmax(tf.add(tf.matmul(cell_output, output_ws), output_bs)), output_bs = zeros, for now
		with tf.variable_scope("vanLSTM_decoder/decoder_accessory"):
			self.output_ws = tf.get_variable("output_ws", [args.hidden_size, args.output_vocab_size])
			output_converter_lambda = lambda cell_output_: tf.nn.softmax(logits=tf.matmul(cell_output_, self.output_ws), dim=1)
			self.output_converter_lambda = output_converter_lambda

		# TODO: (improve) Multi-layer RNN ocnstruction, if more than one layer
		# encoder_cell = rnn_cell.MultiRNNCell([encoder_cell] * args.num_layers, state_is_tuple=True)
		# decoder_cell = rnn_cell.MultiRNNCell([decoder_cell] * args.num_layers, state_is_tuple=True)
		
		# TODO: (improve) Dropout layer can be added here
		# Store the recurrent unit
		self.encoder_cell = encoder_cell
		self.decoder_cell = decoder_cell

		# Input data contains sequences of input tokens of input_embedding_size dimension
		self.input_data = tf.placeholder(tf.float32, [None, args.seq_length, args.input_embedding_size])
		# Target data contains sequences of output tokens of output_vocab_size dimension
		self.target_data = tf.placeholder(tf.float32, [None, args.seq_length, args.output_vocab_size])

        # Learning rate
		self.lr = tf.Variable(args.learning_rate, trainable=False, name="learning_rate")

		# Initial cell state of LSTM (initialized with zeros)
		# TODO: (improve) might use xavier initializer? There seems to be no a staright-forward way
		self.initial_state = encoder_cell.zero_state(batch_size=args.batch_size, dtype=tf.float32)

		# Split inputs according to sequences: a 3D Tensor, num_of_seq x seq_length x input_embedding_size
		# -> list of size seq_length, each of whose element is of num_of_seq x 1 x input_embedding_size
		if tf.__version__[0:2] == '0.':
			input_data_temp = tf.split(split_dim=1, num_split=args.seq_length, value=self.input_data)
		elif tf.__version__[0:2] == '1.':
			input_data_temp = tf.split(value=self.input_data, num_split=args.seq_length, split_dim=1)
		# list of size seq_length, each of which is num_of_seq x 1 x input_embedding_size
		# -> list of size seq_length, each of which is num_of_seq x input_embedding_size
		input_data_list = [tf.squeeze(input=inputs_member, axis=[1]) for inputs_member in input_data_temp]
		del input_data_temp
		
		## This is where the LSTM models differ from each other in substance.
		## The other code might also differ but they are not substantial.
		# call the encoder
		#print("[DEBUG] self.initial_state: " + str(self.initial_state))
		#with tf.variable_scope("vanLSTM_encoder"):
		_, self.encoder_final_state = rnn_segment.run(cell=encoder_cell, 
													  inputs=input_data_list, 
													  initial_state=self.initial_state, 
													  feed_previous=False,
													  loop_func=None,
													  scope="vanLSTM_encoder")
		# call the decoder
		#with tf.variable_scope("vanLSTM_decoder"):
		#print("[DEBUG] self.encoder_final_state: " + str(self.encoder_final_state))
		#print("[DEBUG] self.decoder_inital_state: " + str(self.decoder_cell.zero_state(batch_size=args.batch_size, dtype=tf.float32)))
		self.output_data = self.input_data
		self.cell_outputs, _ = rnn_segment.run(cell=decoder_cell, 
											  inputs=input_data_list, 
											  initial_state=self.encoder_final_state, 
											  feed_previous=True,
											  loop_func=self.output_converter_lambda,
											  scope="vanLSTM_decoder")
		self.output_data = [output_converter_lambda(cell_output_) for cell_output_ in self.cell_outputs]
		
		def get_sum_of_cost(cell_outputs, targets):
			'''
			Calculate the sum of cost of this training batch using cross entropy.
			params:
			cell_outputs: list of length seq_length, each element of size batch_size x hidden_size
			targets: list of length seq_length, each element of size batch_size x output_vocab_size (one-hot)
			                                          # batch_size x seq_length x output_vocab_size (one-hot)
			'''
			# affine mapping from hidden_size dimensional space to output_vocab_size dimensional space
			# affine_mapped_cell_outputs is list of lengrh seq_length, each of size batch_size x output_vocab_size
			affine_mapped_cell_outputs = [tf.matmul(cell_output_, self.output_ws) for cell_output_ in cell_outputs]
			# affine_mapped_cell_outputs is NOT softmaxed.
			# affine_mapped_cell_outputs and targets are both list of length seq_length, 
			# and their elements are of size batch_size x output_vocab_size
			sum_of_cost = 0.
			# for i in xrange(len(targets)):
			# 	affined_mapped_ = affine_mapped_cell_outputs[i]
			# 	target_ = targets[i]
			# 	sum_of_cost += tf.nn.softmax_cross_entropy_with_logits(logits=affined_mapped_, labels=target_, name=None).eval()
			return sum_of_cost


		# Compute the cost scalar: specifically, the average cost per sequence
		sum_of_cost = get_sum_of_cost(cell_outputs=self.cell_outputs, targets=self.target_data)
		#self.cost = tf.Variable(0.)
		self.cost = tf.div(sum_of_cost, args.batch_size)
		print("\n[DEBUG] self.cost: ") + str(self.cost)

		# Get trainable_variables list, print them, and compute the gradients
		# Also clip the gradients if they are larger than args.grad_clip
		trainable_vars = tf.trainable_variables()
		num_trainable_components = 0
		print("\nNumber of trainable Tensors = " + str(len(trainable_vars)))
		for i, var in enumerate(trainable_vars):
			num_trainable_components += np.product(trainable_vars[i].get_shape().as_list())
			print(" " + str(trainable_vars[i].name) + \
			      "\t"  + str(trainable_vars[i].get_shape()) + \
				  " x " + str(trainable_vars[i].dtype.name))
		print("Number of trainable scalar components = " + str(num_trainable_components) + "\n")
		self.num_of_trainable_components = num_trainable_components

		# self.gradients is a list of tuples of (grad_value, variable_name)
		self.gradients = tf.gradients(self.cost, trainable_vars)
		if args.model_unit_test_flag == True:
			for i in xrange(len(self.gradients)):
				if self.gradients[i] == None:
					self.gradients[i] = tf.zeros(shape=trainable_vars[i].get_shape(), dtype=tf.float32)

		if args.grad_clip != None:
			clipped_grads, _ = tf.clip_by_global_norm(self.gradients, args.grad_clip)
		else:
			clipped_grads = self.gradients

		# Using RMSprop, inspired by the LSTM paper of Dr. Alahi, Prof. Saverese, and Prof. Fei-Fei Li
		if args.optimizer_choice == "rms":
			optimizer = tf.train.RMSPropOptimizer(self.lr)
		elif args.optimizer_choice == "adam":
			optimizer = tf.train.AdamOptimizer(self.lr)
		elif args.optimizer_choice == "grad_desc":
			optimizer = tf.train.GradientDescentOptimizer(self.lr)
		else:
			raise ValueError("Optimizer not supported: " + args.optimizer_choice)

		# Train operator. Apply gradients. If a gradient of a variable is None, it will be weeded out.
		self.train_op = optimizer.apply_gradients(zip(clipped_grads, trainable_vars))

def main():
	'''
	This function is used for unit testing this module.
	    args.hidden_size (i.e. the size of hidden state)
		args.num_layers (default = 1, i.e. no stacking), 
		args.seq_length, 
	    args.input_embedding_size, 
		args.output_vocab_size,
		args.batch_size (i.e. the number of sequences in each batch)
		args.optimizer_choice (defualt = "rms", also could be "adam", "grad_desc")
		args.learning_rate, 
		args.grad_clip
	'''
	parser = argparse.ArgumentParser()
	# RNN cell hidden state's size
	parser.add_argument('--hidden_size', type=int, default=96,
	                    help='size of RNN cell hidden state')
	# Number of stacked RNN layers. Only a single layer implemented
	parser.add_argument('--num_layers', type=int, default=1,
	                    help='number of stacked RNN layers')
	# Maximum length of each sequence
	parser.add_argument('--seq_length', type=int, default=20,
	                    help='maximum length of each sequence')
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
	# Model unit testing flag
	parser.add_argument('--model_unit_test_flag', type=bool, default=True,
	                    help='only set to true when performing unit test')
	# Parse the arguments, and construct the model
	args = parser.parse_args()
	args.model_unit_test_flag = True
	model = VanillaLSTMModel(args)
	# TODO: (improve) maybe we do something more?


if __name__ == "__main__":
	main()
	print("this module is functioning.") # test passed. March 5, 2017