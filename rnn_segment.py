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

def rnn_segment_run(cell, inputs, initial_state, feed_previous=False):
	'''
	RNN segment works.
	Params:
	    cell: RNN cell, created by tf.nn.rnn_cell.BasicLSTMCell or else.
	    inputs: list of tensors, variable length, each element is of size 
	        batch_size x embedding_size
	    initial_state: the initial state, of size batch_size x hidden_size
	    feed_previous: if True, then a cell's input is the previous cell's
	        output, with the exception of the first cell, whose input is
		    the first element of the input variable: inputs.
	 Returns:
	     outputs: list of tensors, length equals to the inputs, each
		     element is of size batch_size x cell_output_size
	'''
	if inputs == []:
		raise ValueError('RNN segment inputs should not be an empty list')
	
	state = initial_state
	outputs = []
	
	if feed_previous == False:
		for input_ in inputs:
			output, state = cell(input_, state)
			outputs.append(output)
	elif feed_previous == True:
		output, state = cell(inputs[0],state)
		outputs.append(output)
		if len(inputs) >= 2:
			for input_ in inputs[1:]:
				output, state = cell(outputs[-1], state)
				outputs.append(output)
	else:
		raise ValueError('feed_previous is not a boolean')
	
	return (outputs, state)