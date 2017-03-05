"""
RNN segment functions
CS224N Project

Author: Haihong Li
Date: March 3, 2017

RNN segment: a train of RNN cells (stacked vertically or single layer) linked together in series.
              o u t p u t s
              | | |     |
initial_state-o-o-o-...-o- final_state
              | | |     |
              i n p u t s
"""

import tensorflow as tf
import numpy as np
import ipdb
from tensorflow.python.util import nest

def run(cell, inputs, initial_state, feed_previous=False, scope="rnn"):
	'''
	RNN segment works.
	Params:
	    cell: RNN cell, created by tf.nn.rnn_cell.BasicLSTMCell or else.
	    inputs: list of Tensors, variable length, each element is of size 
	        batch_size x embedding_size
	    initial_state: the initial state, of size batch_size x hidden_size
	    feed_previous: if True, then a cell's input is the previous cell's
	        output, with the exception of the first cell, whose input is
		    the first element of the input variable: inputs.
		scope: VariableScope for the created subgraph
	 Returns:
	     outputs: list of tensors, length equals to the inputs, each
		     element is of size batch_size x cell_output_size
		 state: the hidden state in the end of this cell segment
	'''

	state = initial_state
	outputs = []
	#scope = "rnn"

	#outputs, state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)
	#return (outputs, state)
	print "run() called"
	with tf.variable_scope(scope):
		cell_state = initial_state
		outputs = []
		prev_cell_output = None
		for i, cell_input in enumerate(inputs):
			if (feed_previous == True) and (prev_cell_output != None):
				cell_input = prev_cell_output
			if i > 0:
				tf.get_variable_scope().reuse_variables()
			cell_output, cell_state = cell(cell_input, cell_state)
			outputs.append(cell_output)
			if feed_previous == True:
				prev_cell_output = cell_output
	return outputs, cell_state


if __name__ == "__main__":
	cell = tf.nn.rnn_cell.BasicLSTMCell(3)
	initial_state = cell.zero_state(batch_size=2, dtype=tf.float32)
	inputs = np.array([tf.constant([[1.,2.,3.],[4.,5.,6.]]),tf.constant([[7.,8.,9.],[10.,11.,12.]])])
	with tf.Session() as sess:
		with tf.variable_scope('test_1'):
			run(cell,inputs,initial_state,feed_previous=False)
		with tf.variable_scope('test_2'):
			run(cell,inputs,initial_state,feed_previous=True)
		print("this module is functioning.") # test passed. March 5, 2017