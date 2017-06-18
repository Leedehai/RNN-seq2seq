"""
RNN segment functions implemented with TensorFlow 0.12.1, Python 2.7.12
- - - - -
CS224N Project at Stanford Univeristy
Project mentor: Prof. Chirs Manning

Author: Haihong Li
Date: March 1, 2017
- - - - -
RNN segment: a train of RNN cells (stacked vertically or single layer) linked together in series.
                o  u  t  p  u  t  s
                   |   |   |   |
    initial_state -o - o - o - o - final_state
                   |   |   |   |
                 i  n  p  u  t  s
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
     (3) TensorFlow's basic RNN cell, GRU cell, LSTM cell all apply.
"""
# Libraries
import tensorflow as tf
import numpy as np

# Custom modules
from verbose_print import vprint

class RNNChain():
    '''
    An RNN segment is comprised of multiple RNN cells linked in one-dimensional series.
    If used in stacked multi-layerd RNN, then the cell itself should be multilayered, and
    TensorFlow has a function to create such a cell: tf.nn.rnn_cell.MultiRNNCell().
    '''
    def __init__(self, cell, scope="rnn_segment", name="unnamed"):
        '''
        Constructor.
        Params:
            cell: RNN cell, created by tf.nn.rnn_cell.BasicLSTMCell() or else.
            scope: VariableScope for the created subgraph, set to "rnn_segment" by default
            name: name of this segment, set to "unnamed" by default
        Returns:
            None
        '''
        self.cell = cell
        self.scope = scope
        self.name = name
        self.outputs = None
        self.final_state = None


    def get_cell(self):
        return self.cell

    def get_scope(self):
        return self.scope
    
    def get_name(self):
        return self.name
    
    def run(self, inputs, chain_length, initial_state, cell_input_size=None, feed_previous=False, loop_func=None, verbose=False):
        '''
        RNN segment works.
        Params:
            inputs: list of Tensors, variable length, each element is of size 
                batch_size x input_vector_size. If feed_previous == True, then
                inputs does nothing. inputs should not be None, because it has
                other delicate uses later in this code.
            chain_length: length of this RNN chain. If feed_previous=True, then
                chain_length does nothing (can be None) because the length of 
                chain is determined by inputs' length; if False, then chain_length 
                determines the length of this RNN chain.
            initial_state: the initial state, of size batch_size x hidden_size
            cell_input_size: the size of each cell's input, can be either None, or 
                a 2-integer list [batch_size, input_vector_size]. If cell_input_size 
                is None or feed_previous=False, then cell_input_size assumes the 
                value of the size of the input acquired by the first cell, regardless 
                of what cell_input_size is.
            feed_previous: if True, then a cell's input is the previous cell's
                output processed by the loop affine function, except the first cell, 
                whose input is specified in code with size of cell_input_size (i.e. 
                [batch_size, input_vector_size] 2-integer list).
            loop_func: a lambda function that converts a cell's output from 
                batch_size x hidden_size to batch_size x output_vocab_size
            verbose: verbosity flag
        Returns:
            outputs: list of tensors, length equals to the inputs, each element is 
                of size batch_size x hidden_size. NOTE NOT converted to yhat
            cell_state: the cell state in the end of this cell segment
        '''
        if inputs == None:
            raise ValueError("RNNChain::run()'s inputs should not be None")
        if feed_previous == True and loop_func == None:
            raise ValueError("feed_previous is True, but loop_func is not given")
        if feed_previous == True:
            # This is a hack. Reassign inputs.
            inputs = range(chain_length)

        cell = self.cell
        scope = self.scope
        state = initial_state
        outputs = []
        
        # if verbose: 
        # 	print "\n\033[32m[INFO] an rnn_segement.run() is linked into the computational graph in scope " + scope + "\033[m"
        # 	print "\n[INFO] an rnn_segement.run() is linked into the computational graph in scope " + scope
        vprint(verbose, "\n[INFO] an rnn_segement.run() is linked into the computational graph in scope " + scope, color="g")

        with tf.variable_scope(scope):
            cell_state = initial_state
            vprint(verbose, self.get_info_str(cell_state))
            outputs = []
            if feed_previous == True:
                if cell_input_size == None or isinstance(cell_input_size,list) == False or len(list(cell_input_size)) != 2:
                    raise ValueError("cell_input_size should be a two-integer list, [batch_size, input_vector_size]")
                # TODO: (improve) assume the input to the first cell is a tensor of zero, for now
                prev_cell_output_yhat = tf.zeros(list(cell_input_size))
            for i, cell_input in enumerate(inputs):
                if (feed_previous == True) and (prev_cell_output_yhat != None):
                    # in this case, ignore cell_input's value and reassign it
                    cell_input = prev_cell_output_yhat
                if i > 0:
                    # if the cell is reused once, declare reusing since the second time you use it
                    tf.get_variable_scope().reuse_variables()
                vprint(verbose,
                       "\n[INFO] BEFORE CELL " + scope + "\n" + str(i)
                       + " cell_input: Tensor " + str(cell_input.get_shape().as_list()),
                       color="b")
                vprint(verbose, self.get_info_str(cell_state), color="b")

                # cell_output: batch_size x hidden_size, cell_state's dimension depends on cell's type
                cell_output, cell_state = cell(cell_input, cell_state)
                vprint(verbose,
                       "[INFO] AFTER CELL " + scope + "\n" + str(i) 
                       + " cell_output: Tensor " + str(cell_output.get_shape().as_list()),
                       color="cyan")
                vprint(verbose, self.get_info_str(cell_state), color="cyan")

                # append the cell's output to the output sequence
                outputs.append(cell_output)

                if feed_previous == True:
                    prev_cell_output_yhat = loop_func(cell_output)
                    vprint(verbose,
                           "\n[INFO] PREV_CELL_OUTPUT " + scope + "\n" + str(i) 
                           + " prev_cell_output_yhat: " + str(prev_cell_output_yhat.get_shape().as_list()),
                           color="YELLOW")
                # the last cell's state is the final state
                final_state = cell_state

        # NOTE "outputs" are the cells' immediate outputs, not converted to yhat.
        self.outputs = outputs
        self.final_state = final_state
        return outputs, final_state

    def get_info_str(self, cell_state):
        '''
        Handling different types of cell_state and returns its shape. cell_state's type depends on cell's type.
        Params:
            cell_state
        Returns:
            a string describing the shape of cell_state.
        '''
        try:
			# A hack: depending on the cell, cell_state may be a tuple of Tensors (like LSTM)
			return str([tp.get_shape().as_list() for tp in cell_state])
        except:
			pass

        try:
			# A hack: depending on the cell, cell_state may be a tuple of tuples of Tensors (like stacked LSTM)
			return str([tp[0].get_shape().as_list() for tp in cell_state]) + " x " + str(len(cell_state)) + " layers."
        except:
			pass

        try:
			# A hack: depending on the cell, cell_state may be a Tensor (like GRU)
			return str(cell_state.get_shape().as_list())
        except:
			pass

        try:
			# A hack: depending on the cell, cell_state may be a tuple of Tensors (like stacked GRU)
			return str(cell_state[0].get_shape().as_list() for tp in cell_state) + " x " + str(len(cell_state)) + " layers."
        except:
			pass

        raise TypeError("cell_state is not analyzable.")

if __name__ == "__main__":
    # Single-layer LSTM cell
	cell1 = tf.nn.rnn_cell.BasicLSTMCell(3)
	initial_state1 = cell1.zero_state(batch_size=2, dtype=tf.float32)
    # Double-layer LSTM cell
	cell2 = tf.nn.rnn_cell.MultiRNNCell([cell1] * 2, state_is_tuple=True)
	initial_state2 = cell2.zero_state(batch_size=2, dtype=tf.float32)
	# Single-layer GRU cell
	cell3 = tf.nn.rnn_cell.BasicLSTMCell(3)
	initial_state3 = cell3.zero_state(batch_size=2, dtype=tf.float32)
	# Double-layer GRU cell
	cell4 = tf.nn.rnn_cell.MultiRNNCell([cell3] * 2, state_is_tuple=True)
	initial_state4 = cell4.zero_state(batch_size=2, dtype=tf.float32)

	inputs = np.array([tf.constant([[1.,2.,3.],[4.,5.,6.]]),tf.constant([[7.,8.,9.],[10.,11.,12.]])])
	with tf.Session() as sess:
		with tf.variable_scope('test_cell1_1'):
			rs = RNNChain(cell1)
			rs.run(inputs, initial_state1, cell_input_size=[2,3], feed_previous=False)
			print rs.get_name()
		with tf.variable_scope('test_cell1_2'):
			rs = RNNChain(cell1)
			rs.run(inputs, initial_state1, cell_input_size=[2,3], feed_previous=True, loop_func=lambda x : x)
			print rs.get_scope()
		with tf.variable_scope('test_cell2'):
			rs = RNNChain(cell2, name='rs_3', scope='test_3')
			rs.run(inputs, initial_state2, cell_input_size=[2,3], feed_previous=True, loop_func=lambda x : x)
			print rs.final_state
		with tf.variable_scope('test_cell3'):
			rs = RNNChain(cell3, name='rs_4', scope='test_4')
			rs.run(inputs, initial_state3, cell_input_size=[2,3], feed_previous=True, loop_func=lambda x : x)
			print rs.final_state
		with tf.variable_scope('test_cell4'):
			rs = RNNChain(cell4, name='rs_5', scope='test_5')
			rs.run(inputs, initial_state4, cell_input_size=[2,3], feed_previous=True, loop_func=lambda x : x)
			print rs.final_state
        print("\nThis module is functioning.") # test passed. 00:49 March 10, 2017