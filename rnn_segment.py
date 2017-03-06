"""
RNN segment functions
- - - - -
CS224N Project at Stanford Univeristy
Project mentor: Prof. Chirs Manning

Author: Haihong Li
Date: March 1, 2017
- - - - -
RNN segment: a train of RNN cells (stacked vertically or single layer) linked together in series.
                   o u t p u t s
                   | | |     |
    initial_state -o-o-o-...-o- final_state
                   | | |     |
                   i n p u t s
inputs is a list of tensors, each of which is batch_size input vectors stacked together:
    <i n p u t _ v e c t o r _ 1>
	<i n p u t _ v e c t o r _ 2>
	     ...................                        
	<i n p u t _ v e c t o r _ n>   (n = batch_size)
outputs is a list of tensors, each of which is batch_size output vectors stacked together
in a similar fashion.
NOTE outputs are cells' orginal outputs. For LSTM, they are hidden states, NOT predictions (yhat).
NOTE This module does not specify the type of RNN cell. It only requires
     (1) cells in the same segment are homogeneous, and 
	 (2) a cell works in this way: cell_output, cell_state = cell(cell_input, previous_cell_state)
"""

import tensorflow as tf
import numpy as np

import pdb

def run(cell, inputs, initial_state, cell_input_size=None, feed_previous=False, loop_func=None, scope="rnn_segment", verbose=False):
	'''
	RNN segment works.
	Params:
	    cell: RNN cell, created by tf.nn.rnn_cell.BasicLSTMCell or else.
	    inputs: list of Tensors, variable length, each element is of size 
	        batch_size x input_vector_size. If feed_previous == True, then
			inputs does nothing other than determining the length of this
			RNN segment. inputs should not be None.
	    initial_state: the initial state, of size batch_size x hidden_size
		cell_input_size: the size of each cell's input, can be either None, or 
		    a 2-integer list [batch_size, input_vector_size]. If cell_input_size 
			is None or feed_previous=False, then cell_input_size assumes the 
			value of the size of the input acquired by the first cell, regardless 
			of what cell_input_size is.
	    feed_previous: if True, then a cell's input is the previous cell's
	        output processed by loop_func, with the exception of the first cell, 
			whose input is specified in code with size of cell_input_size (i.e. 
			[batch_size, input_vector_size] 2-integer list).
		loop_func: explained above. It convert batch_size x hidden_size to 
		    batch_size x output_vocab_size
		scope: VariableScope for the created subgraph
	 Returns:
	     outputs: list of tensors, length equals to the inputs, each
		     element is of size batch_size x hidden_size
			 NOTE NOT converted to yhat
		 state: the hidden state in the end of this cell segment
	'''
	if feed_previous == True and loop_func == None:
		raise ValueError("feed_previous is True, but loop_func is not given")

	state = initial_state
	outputs = []
	
	if verbose: 
		print "\n\033[32m[INFO] an rnn_segement.run() is linked into the computational graph in scope " + scope + "\033[m"
		print "\n[INFO] an rnn_segement.run() is linked into the computational graph in scope " + scope
	
	with tf.variable_scope(scope):
		cell_state = initial_state
		if verbose: 
			print("\n[INFO] rnn_segment initial_state: " + str([tp.get_shape().as_list() for tp in cell_state]))
		outputs = []
		if feed_previous == True:
			if cell_input_size == None or isinstance(cell_input_size,list) == False or len(list(cell_input_size)) != 2:
				raise ValueError("cell_input_size should be a two-integer list, [[batch_size, input_vector_size]]")
			# TODO: (improve) assume the input to the first cell is a tensor of zero, for now
			prev_cell_output_yhat = tf.zeros(list(cell_input_size))
		for i, cell_input in enumerate(inputs):
			if (feed_previous == True) and (prev_cell_output_yhat != None):
				cell_input = prev_cell_output_yhat
			if i > 0:
				tf.get_variable_scope().reuse_variables()
			if verbose:
				print("\n[INFO] \033[34mBEFORE CELL " + scope + "\n" + str(i) + " \033[mcell_input: Tensor " + str(cell_input.get_shape().as_list()))
				print("[INFO] \033[34mBEFORE CELL " + scope + "\n" + str(i) + " \033[mcell_state: Tensor tuple " + str([tp.get_shape().as_list() for tp in cell_state]))
			cell_output, cell_state = cell(cell_input, cell_state)
			if verbose:
				print("[INFO] \033[36mAFTER CELL " + scope + "\n" + str(i) + " \033[mcell_output: Tensor " + str(cell_output.get_shape().as_list()))
				print("[INFO] \033[36mAFTER CELL " + scope + "\n" + str(i) + " \033[mcell_state: Tensor tuple " + str([tp.get_shape().as_list() for tp in cell_state]))
			outputs.append(cell_output)
			if feed_previous == True:
				prev_cell_output_yhat = loop_func(cell_output)
				if verbose:
					print("\n[INFO] \033[33mPREV_CELL_OUTPUT " + scope + "\n" + str(i) + " \033[mprev_cell_output_yhat: " + str(prev_cell_output_yhat.get_shape().as_list()))
	# NOTE outputs not converted to yhat.
	return outputs, cell_state

if __name__ == "__main__":
	cell = tf.nn.rnn_cell.BasicLSTMCell(3)
	initial_state = cell.zero_state(batch_size=2, dtype=tf.float32)
	inputs = np.array([tf.constant([[1.,2.,3.],[4.,5.,6.]]),tf.constant([[7.,8.,9.],[10.,11.,12.]])])
	with tf.Session() as sess:
		with tf.variable_scope('test_1'):
			run(cell, inputs, initial_state, cell_input_size=[2,3], feed_previous=False)
		with tf.variable_scope('test_2'):
			run(cell, inputs, initial_state, cell_input_size=[2,3], feed_previous=True, loop_func=lambda x : x)
		print("this module is functioning.") # test passed. 20:55 March 5, 2017