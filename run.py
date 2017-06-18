"""
Top level training script implemented with TensorFlow 0.12.1, Python 2.7.12
- - - - -
CS224N Project at Stanford Univeristy
Project mentor: Prof. Chirs Manning

Author: Haihong (@Leedehai)
Date: March 7, 2017
- - - - -
This is the top level training script that calls upon other code.
"""
# Libraries
import argparse

# Custom modules
from verbose_print import vprint
from Dataloader import Dataloader
from Trainer import Trainer
from VanillaLSTMTransModel import VanillaLSTMTransModel
#from HierLSTMTransModel import HierLSTMTransModel
#from AttenHierLSTMTransModel import AttenHierLSTMTransModel
from Tester import Tester

### Argument setting.
# VanillaLSTMTransModel
parser = argparse.ArgumentParser()

############################## RNN Configuration ##############################
# RNN cell hidden state's size
parser.add_argument('--hidden_size', type=int, default=5,
	                    help='size of RNN cell hidden state') # Do NOT Touch, write custom values below
# Number of stacked RNN layers. Only a single layer implemented
parser.add_argument('--num_layers', type=int, default=1,
	                    help='number of stacked RNN layers') # Do NOT Touch

################################## Data Info ##################################
# Larger than the maximum number of words in each input sequence
parser.add_argument('--input_seq_length', type=int, default=5,
	                    help='maximum length of each input sequence or larger') # Do NOT Touch
# Larger than the maximum number of words in each target sequence
parser.add_argument('--target_seq_length', type=int, default=6,
	                    help='maximum length of each target sequence or larger') # Do NOT Touch
# Maximum number of sentences in an input sequence (paragraph)
parser.add_argument('--input_num_sent', type=int, default=15,
	                    help='embedding size of input vectors') # Do NOT Touch
# Maximum number of sentences in an output sequence (paragraph)
parser.add_argument('--output_num_sent', type=int, default=15,
	                    help='embedding size of input vectors') # Do NOT Touch
# Embedding size of input
parser.add_argument('--input_embedding_size', type=int, default=50,
	                    help='embedding size of input vectors') # Do NOT Touch
# Embedding size of output
parser.add_argument('--output_vocab_size', type=int, default=400002,
	                    help='size of output vocabulary') # Do NOT Touch
# Target token size, should be 1 because it is word index in vocabulary
parser.add_argument('--target_token_size', type=int, default=1,
	                    help='target token size, normally it should be 1 because it is word index in vocabulary') # Do NOT Touch
############################### Trainer Settings ##############################
# Batch size
parser.add_argument('--batch_size', type=int, default=2,
	                    help='number of sequences in a batch') # Do NOT Touch
# Epochs
parser.add_argument('--epochs', type=int, default=2,
	                    help='Learning rate') # Do NOT Touch	
# Choice of optimzier
parser.add_argument('--optimizer_choice', type=str, default='adam',
	                    help='rms (defualt), adam, grad_desc') # Do NOT Touch
# Learning rate
parser.add_argument('--learning_rate', type=float, default=0.08,
	                    help='Learning rate') # Do NOT Touch	
# Gradient clip, i.e. maximum value of gradient amplitute allowed
parser.add_argument('--grad_clip', type=float, default=None,
	                    help='gradient upbound, i.e. maximum value of gradient amplitute allowed') # Do NOT Touch

############################### Utility Settings ##############################
# Model unit testing flag, default to False
parser.add_argument('-t','--test', action='store_true',
	                    help='only set to true when performing unit test') # Do NOT Touch
# Verbosity flag, default to False
parser.add_argument('-v','--verbose', action='store_true',
	                    help='only set to true when you want verbosity') # Do NOT Touch
# Continuing training flag, default to False
parser.add_argument('-c','--continue_training', action='store_true',
	                    help='if set, then continue training from the previous checkpoint') # Do NOT Touch
# Parse the arguments, and construct the model
args = parser.parse_args()

# You may perform testing without actually loading data by calling: python run.py -t
vprint(True, "run.py -- arg.test = " + str(args.test), color="CYAN")
if args.test == True:
    # Test the code using artificial numbers, without actually loading the data.
	############################## RNN Configuration ##############################
    args.hidden_size = 16
    args.num_layers = 2
	################################## Data Info ##################################
    args.input_seq_length = 21
    args.target_seq_length = 23
    args.input_embedding_size = 16
    args.output_vocab_size = 150
	############################### Trainer Settings ##############################
    args.epochs = 5
    args.batch_size = 8
    args.grad_clip = 15
    args.learning_rate = 0.05
	############################### Utility Settings ##############################
    #args.verbose = True
elif args.test == False:
    ### Write your custom values here to override default values.
	# Keep the default values, for now.
    pass
	############################## RNN Configuration ##############################
    args.hidden_size = 50
    args.num_layers = 1
	################################## Data Info ##################################
    args.input_seq_length = 25
    args.target_seq_length = 25
    args.input_embedding_size = 50
    args.output_vocab_size = 21485
	############################### Trainer Settings ##############################
    args.epochs = 15
    args.batch_size = 32
    args.grad_clip = 15
    args.learning_rate = 0.003
	############################### Utility Settings ##############################
    args.verbose = True

trainer = Trainer(args=args, model_choice='VanillaLSTMTransModel', if_testing=args.test)
trainer.train(num_epochs=args.epochs)