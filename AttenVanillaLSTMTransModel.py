"""
Vanilla LSTM Sequence-to-Sequence Model implemented with TensorFlow 0.12.1, Python 2.7.12
- - - - -
CS224N Project at Stanford Univeristy
Project mentor: Prof. Chris Manning

Author: Haihong (@Leedehai)
Date: March 1, 2017
- - - - -
Vanilla LSTM Sequence-to-Sequence Model, comprised of a classic encoder-decoder model.
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

# The vanilla LSTM model
class AttenVanillaLSTMTransModel():
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
		#if training == False:
		#	args.batch_size = 2
		
		# Store the arguments, and print the important argument values
		self.args = args
		verbose = self.args.verbose
		print("AttenVanillaLSTMTransModel initializer is called..\n" \
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
		# encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.args.hidden_size, state_is_tuple=True)
		# decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.args.hidden_size, state_is_tuple=True)
		encoder_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.args.hidden_size, initializer=tf.contrib.layers.xavier_initializer(), state_is_tuple=True)
		decoder_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.args.hidden_size, initializer=tf.contrib.layers.xavier_initializer(), state_is_tuple=True)
		
		# convert cell's outputs (batch_size x hidden_size for each cell) to batch_size x output_vocab_size
		with tf.variable_scope("vanLSTM_decoder/decoder_accessory"):
			self.attention_wa = tf.get_variable("attention_wa", [2 * self.args.hidden_size, 1])
			self.attention_wc = tf.get_variable("attention_wc", [2 * self.args.hidden_size, self.args.hidden_size])
			self.output_ws = tf.get_variable("output_ws", [self.args.hidden_size, self.args.output_vocab_size])
			output_affine_map_lambda = lambda cell_raw_output_, context_: tf.matmul(tf.tanh(tf.matmul(tf.concat(concat_dim=1, values=[cell_raw_output_, context_]),self.attention_wc)), self.output_ws) # tf.concat() API changed in TensorFlow r1.0
			output_converter_lambda = lambda cell_raw_output_, context_: tf.nn.softmax(logits=output_affine_map_lambda(cell_raw_output_, context_), dim=-1) # -1: last dimension
			self.output_affine_map_lambda = output_affine_map_lambda
			self.output_converter_lambda = output_converter_lambda

		# Multi-layer RNN ocnstruction, if more than one layer
		if self.args.num_layers <= 0 or isinstance(self.args.num_layers, int) == False:
			raise ValueError("Specified number of layers is non-positive or is not an integer.")
		elif self.args.num_layers >= 2:
			vprint(True, "Stacked RNN: number of layers = " + str(self.args.num_layers))
			encoder_cell = tf.nn.rnn_cell.MultiRNNCell([encoder_cell] * self.args.num_layers, state_is_tuple=True)
			decoder_cell = tf.nn.rnn_cell.MultiRNNCell([decoder_cell] * self.args.num_layers, state_is_tuple=True)
		
		# TODO: (improve) Dropout layer can be added here
		# Store the recurrent unit
		self.encoder_cell = encoder_cell
		self.decoder_cell = decoder_cell

		# Create encoder and decoder RNNChain instances
		encoder = RNNChain(self.encoder_cell, name="vanLSTM_decoder", scope="vanLSTM_encoder")
		decoder = RNNChain(self.decoder_cell, name="vanLSTM_decoder", scope="vanLSTM_decoder")
		self.encoder = encoder
		self.decoder = decoder

		# Input data contains sequences of input tokens of input_embedding_size dimension
		self.input_data = tf.placeholder(tf.float32, [None, self.args.input_seq_length, self.args.input_embedding_size])
		# Target data contains sequences of output tokens of target_token_size dimension (=1)
		self.target_data = tf.placeholder(tf.int32, [None, self.args.target_seq_length, self.args.target_token_size])
		# Target lengths list contains numbers of non-padding input tokens in each sequence in this batch, 
		# each element is an integer, indicating the number of non-padding tokens of a sequence.
		self.target_lens_list = tf.placeholder(tf.int32, [None])

        # Learning rate
		self.lr = tf.Variable(self.args.learning_rate, trainable=False, name="learning_rate")

		# Initial cell state of LSTM (initialized with zeros)
		self.initial_state = encoder_cell.zero_state(batch_size=self.args.batch_size, dtype=tf.float32)

		# Preprocessing the information got from placeholders.
		# First, target_lens_list does not need any further actions.
		target_lens_list = self.target_lens_list
		# Second, input_data and target_data need reshaping.
		# Split inputs and targets according to sequences: a 3D Tensor, num_of_seq x seq_length x Di/Vo
		# -> list of size seq_length, each of whose element is of num_of_seq x 1 x Di/Vo
		if tf.__version__[0:2] == '0.':
			input_data_temp = tf.split(split_dim=1, num_split=self.args.input_seq_length, value=self.input_data)
			target_data_temp = tf.split(split_dim=1, num_split=self.args.target_seq_length, value=self.target_data)
		elif tf.__version__[0:2] == '1.':
			input_data_temp = tf.split(value=self.input_data, num_split=self.args.input_seq_length, split_dim=1)
			target_data_temp = tf.split(value=self.target_data, num_split=self.args.target_seq_length, split_dim=1)
		# Squeeze: list of size seq_length, each of which is num_of_seq x 1 x Di/Vo
		# -> list of size seq_length, each of which is num_of_seq x Di/Vo
		input_data_list = [tf.squeeze(input=list_member, axis=[1]) for list_member in input_data_temp]
		target_data_list = [tf.squeeze(input=list_member, axis=[1]) for list_member in target_data_temp]
		del input_data_temp, target_data_temp
		
		## This is where the LSTM models differ from each other in substance.
		## The other code might also differ but they are not substantial.
		# call the encoder
		#print("[DEBUG] self.initial_state: " + str(self.initial_state))
		#with tf.variable_scope("vanLSTM_encoder"):
		vprint(True, "Building encoder...", color="MAG")
		encoder_start_time = time.time()
		# encoder_hidden_states is list of length input_seq_length, each element is batch_size x hidden_size
		self.encoder_hidden_states, self.encoder_final_state = encoder.run(inputs=input_data_list,
		                                          chain_length=None,
												  cell_input_size=[self.args.batch_size, self.args.input_embedding_size],
												  initial_state=self.initial_state, 
												  feed_previous=False,
												  verbose=self.args.verbose)
		self.encoder_end_state = self.initial_state
		encoder_end_time = time.time()
		vprint(True, " -- Encoder built. Time used: " + str(encoder_end_time - encoder_start_time) + " s", color="MAG")

		# call the decoder
		#with tf.variable_scope("vanLSTM_decoder"):
		#print("[DEBUG] self.encoder_final_state: " + str(self.encoder_final_state))
		#print("[DEBUG] self.decoder_inital_state: " + str(self.decoder_cell.zero_state(batch_size=self.args.batch_size, dtype=tf.float32)))
		vprint(True, "Building decoder...", color="MAG")
		decoder_start_time = time.time()
		# cell_raw_outputs is list of length target_seq_length, each element is batch_size x hidden_size, not combined with attention
		self.cell_raw_outputs, _ = decoder.run(inputs=input_data_list,
		                                   chain_length=self.args.target_seq_length,
										   cell_input_size=[self.args.batch_size, self.args.output_vocab_size], 
										   initial_state=self.encoder_final_state, 
										   feed_previous=False,
										   #loop_func=self.output_converter_lambda,
										   verbose=self.args.verbose)
		decoder_end_time = time.time()
		vprint(True, " -- Decoder built. Time used: " + str(decoder_end_time - decoder_start_time) + " s", color="MAG")

		vprint(True, " Building context calculator...", color="MAG")
		context_start_time = time.time()
		# We adopt global attention here. score(hs,ht) = [hs, ht] * attention_wa, hs.T * attention_wa * ht (*: matrix multiplication)
		# attention_wa: 2 * self.args.hidden_size x 1
		# encoder_hidden_states is list of length input_seq_length, each element is batch_size x hidden_size
		self.cell_raw_outputs_and_contexts = []
		for decoder_ht_ in self.cell_raw_outputs:
			score_st = []
			for encoder_hs_ in self.encoder_hidden_states:
				# score_st's each element should be of size batch_size x 1
				print decoder_ht_
				score_st.append(tf.matmul(tf.concat(concat_dim=1,values=[decoder_ht_, encoder_hs_]), self.attention_wa)) # tf.concat() API changed in TensorFlow r1.0
			# alpha_st should be of length input_seq_length, each element of which is of size batch_size x 1
			alpha_st = tf.nn.softmax(score_st, dim=0) # dim = 0: softmaxing for each batch
			context_for_ht_ = tf.zeros([1,self.args.hidden_size])
			for s in xrange(len(self.encoder_hidden_states)):
				encoder_hs_ = self.encoder_hidden_states[s]
				# alpha_st_ is of size batch_size x 1
				alpha_st_ = alpha_st[s]
				# weight_hs is of size batch_size x hidden_size
				weighted_hs_ = tf.mul(alpha_st_, encoder_hs_) # tf.mul() supports broadcasting of dimension
				context_for_ht_ = tf.add(context_for_ht_, weighted_hs_)
			self.cell_raw_outputs_and_contexts.append((decoder_ht_, context_for_ht_))

		context_end_time = time.time()
		vprint(True, " -- Context calculator built. Time used: " + str(context_end_time - context_start_time) + " s", color="MAG")

		vprint(True, "Building output converter...", color="MAG")
		converter_start_time = time.time()
		# output_data is softmaxed. It is a list of length target_seq_length, each element is batch_size x output_vocab_size
		self.output_data = [output_converter_lambda(cell_raw_output_, context_) for (cell_raw_output_,context_) in self.cell_raw_outputs_and_contexts]
		converter_end_time = time.time()
		vprint(True, " -- Converter built. Time used: " + str(converter_end_time - converter_start_time) + " s", color="MAG")

		# Compute the cost scalar: specifically, the average cost per sequence
		vprint(True, "Building cost calculator...", color="MAG")
		self.cell_outputs = [output_affine_map_lambda(cell_raw_output_, context_) for (cell_raw_output_,context_) in self.cell_raw_outputs_and_contexts]
		sum_of_cost = self.get_sum_of_cost(cell_outputs=self.cell_outputs, targets=target_data_list, targets_lens=target_lens_list)
		#self.cost = tf.Variable(0.)
		self.cost = tf.div(sum_of_cost, self.args.batch_size)
		print("\n[DEBUG] self.cost: ")
		print self.cost

		# We only deal with back-propagration during training phase.
		if training == True:
			# Get trainable_variables list and count them.
			# Also clip the gradients if they are larger than self.args.grad_clip
			vprint(True, "\nAggregating all trainable variables...", color="BLUE")
			trainable_vars = tf.trainable_variables()
			num_trainable_components = 0
			vprint(True, "\nNumber of trainable Tensors = " + str(len(trainable_vars)), color="GREEN")
			for i, var in enumerate(trainable_vars):
				num_trainable_components += np.product(trainable_vars[i].get_shape().as_list())
				vprint(True,
					" " + str(trainable_vars[i].name) + \
					"\t"  + str(trainable_vars[i].get_shape()) + \
					" x " + str(trainable_vars[i].dtype.name),
					color="GREEN")
			vprint(True,
				"Number of trainable scalar components = " + str(num_trainable_components),
				color="GREEN")
			if num_trainable_components >= 1e3 and num_trainable_components < 1e4:
				vprint(True, " -- that is in the order of 10e3: thousands\n", color="GREEN")
			elif num_trainable_components >= 1e4 and num_trainable_components < 1e5:
				vprint(True, " -- that is in the order of 10e4: tens of thousands\n", color="GREEN")
			elif num_trainable_components >= 1e5 and num_trainable_components < 1e6:
				vprint(True, " -- that is in the order of 10e5: hundreds of thousands\n", color="GREEN")
			elif num_trainable_components >= 1e6 and num_trainable_components < 1e7:
				vprint(True, " -- that is in the order of 10e6: millions\n", color="GREEN")
			elif num_trainable_components >= 1e7 and num_trainable_components < 1e8:
				vprint(True, " -- that is in the order of 10e7: tens of millions\n", color="GREEN")
			elif num_trainable_components >= 1e8 and num_trainable_components < 1e9:
				vprint(True, " -- that is in the order of 10e8: hundreds of millions\n", color="GREEN")
			elif num_trainable_components >= 1e9:
				vprint(True, " -- that is in the order of 10e9 to 10e-Infinity: billions or higher", color="GREEN")
			self.num_of_trainable_components = num_trainable_components

			# Compute the gradient of cost with respect of the trainable variables.
			vprint(True, "Calculating gradient expressions for all trainable variables. Be patient...", color="BLUE")
			grad_start_time = time.time()
			# self.gradients is a list of tuples of (grad_value, variable_name)
			self.gradients = tf.gradients(self.cost, trainable_vars)
			grad_end_time = time.time()
			vprint(True,
				" -- Finished calculating gradient expressions. Time used: " + str(grad_end_time - grad_start_time) + " s",
				color="BLUE")
			# A hack: when testing, elements in gradients may ALL be None, and it causes problems in clip_by_global_norm()
			
			# This is just for validation of the code.
			if self.args.test == True:
				print("TESTING TESTING TESTING")
				for i in xrange(len(self.gradients)):
					if self.gradients[i] == None:
						self.gradients[i] = tf.zeros(shape=trainable_vars[i].get_shape(), dtype=tf.float32)

			if self.args.grad_clip != None:
				clipped_grads, _ = tf.clip_by_global_norm(self.gradients, self.args.grad_clip)
			else:
				clipped_grads = self.gradients

			# Using RMSprop, inspired by the LSTM paper of Dr. Alahi, Prof. Saverese, and Prof. Fei-Fei Li
			if self.args.optimizer_choice == "rms":
				optimizer = tf.train.RMSPropOptimizer(self.lr)
			elif self.args.optimizer_choice == "adam":
				optimizer = tf.train.AdamOptimizer(self.lr)
			elif self.args.optimizer_choice == "grad_desc":
				optimizer = tf.train.GradientDescentOptimizer(self.lr)
			else:
				raise ValueError("Optimizer not supported: " + self.args.optimizer_choice)

			# Train operator. Apply gradients. If a gradient of a variable is None, it will be weeded out.
			self.train_op = optimizer.apply_gradients(zip(clipped_grads, trainable_vars))
	
    def get_sum_of_cost(self, cell_outputs, targets, targets_lens):
		'''
		Calculate the sum of cost of this training batch using cross entropy.
		This function takes padding information into account, leading to existence of hacks.
		NOTE that padding target word's token are all 0.
		params:
			cell_outputs: list of length input_seq_length, each element of which is of size batch_size x hidden_size
			targets: list of length target_seq_length, each element of which is of size batch_size x target_token_size (=1)
			targets_lens: list of length batch_size, each element of which is an integer, indicating the "non-padding" length of each sequence.
		'''
		args = self.args

		# First, deal with cell_outputs.
		# mapping from hidden_size dimensional space to output_vocab_size dimensional space
		# cell_outputs is a list of length target_seq_length, each element of size batch_size x hidden_size
		# affine_mapped_cell_outputs is a list of length target_seq_length, each element of size batch_size x output_vocab_size
		# affine_mapped_cell_outputs is NOT softmaxed.
		one_arg_affine_map_lambda = lambda cell_output_: tf.matmul(cell_output_, self.output_ws)
		affine_mapped_cell_outputs = [one_arg_affine_map_lambda(cell_raw_output_) for cell_raw_output_ in cell_outputs]
		# transformed_amco is of size batch_size x target_seq_length x output_vocab_size
		# "amco" stands for "affine_mapped_cell_outputs"
		transformed_amco = tf.transpose(affine_mapped_cell_outputs, perm=[1,0,2])

		# Second, deal with targets.
		# Note that targets is a list of length target_seq_length, each element of which is of size batch_size x target_token_size (=1)
		# It need to be transformed into an array of size batch_size x target_seq_length, and each atom element is an integer.
		# targets is first squeezed_targets into a tensor of size target_seq_length x batch_size, and then transposed.
		# transformed_targets is of size batch_size x target_seq_length, and each atom element is an integer.
		transformed_targets = tf.transpose(tf.squeeze(input=targets, axis=2), perm=[1,0])
			
		# Third, deal with targets_lens.
		# batch_seq_mask is a tensor of shape batch_size x target_seq_length
		print targets_lens
		batch_seq_mask = tf.sequence_mask(lengths=targets_lens, maxlen=self.args.target_seq_length)
		# masked_targets is a tensor of shape ([?],). Its length is the sum of all non-padding tokens in this target batch.
		masked_targets = tf.boolean_mask(tensor=transformed_targets, mask=batch_seq_mask)
			
		# Forth, mask the cell_outputs as well. I made some compromise here.
		# amco stands for "affine_mapped_cell_outputs"
		# masked_amco is a tensor of shape ([?],). Its length is the sum of all non-padding tokens in this target batch.
		masked_amco = tf.boolean_mask(tensor=transformed_amco, mask=batch_seq_mask)

		sum_of_cost = 0.0 # keep this statement for convenience of debugging
		sum_of_cost = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=masked_amco, labels=masked_targets))
		# sum_of_cost = 0.0
		# # targets is a list of length target_seq_length, and each element is of size batch_size x target_token_size (=1)
		# for i in xrange(len(targets)):
		# 	# affine_mapped_ is of size batch_size x output_vocab_size, representing probability distributions over output vocabulary
		# 	affine_mapped_ = affine_mapped_cell_outputs[i]
		# 	# target_ is of size batch_size, NOT batch_size x 1
		# 	target_ = tf.squeeze(input=targets[i], axis=[1])
		# 	# target_len_ is an integer
		# 	target_len_ = targets_lens[i]
		# 	sum_of_cost += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=affine_mapped_, labels=target_))
		return sum_of_cost
	
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
		args.target_token_size
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
	parser.add_argument('--input_embedding_size', type=int, default=6,
	                    help='embedding size of input vectors')
	# Output vocabulary
	parser.add_argument('--output_vocab_size', type=int, default=8,
	                    help='size of output vocabulary')
	# Target token size, should be 1 because it is word index in vocabulary
	parser.add_argument('--target_token_size', type=int, default=1,
	                    help='target token size, normally it should be 1 because it is word index in vocabulary')
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
	model = AttenVanillaLSTMTransModel(args)
	print "arguments:"
	vprint(args.verbose, model.get_args().__dict__, color=None)
	# TODO: (improve) maybe we do something more?

if __name__ == "__main__":
	main()
	print("This module: " + __file__ + " is functioning.") # test passed. 23:06 March 5, 2017