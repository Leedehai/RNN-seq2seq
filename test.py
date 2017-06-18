"""
Test script implemented with TensorFlow 0.12.1, Python 2.7.12
- - - - -
CS224N Project at Stanford Univeristy
Project mentor: Prof. Chirs Manning

Author: Haihong (@Leedehai)
Date: March 7, 2017
- - - - -
This is the test script that calls upon other code.
"""
# Libraries
import argparse
import pickle
import os

# Custom modules
from verbose_print import vprint
from Dataloader import Dataloader
from Trainer import Trainer
from VanillaLSTMTransModel import VanillaLSTMTransModel
#from HierLSTMTransModel import HierLSTMTransModel
#from AttenHierLSTMTransModel import AttenHierLSTMTransModel
from Tester import Tester

# Which model you would like to test? Maybe I should make it a commandline argument..
model_abbr = "V"

# The directory in which generated files reside
directory = "../RUN_" + model_abbr
try:
    with open(os.path.join(directory, 'args.pkl'), 'r+') as f:
        saved_args = pickle.load(f)
        # Don't forget this line below. It tells the model that rebuilding it is not for training.
        saved_args.continue_training = False
except Exception as e:
    print e
    raise ValueError("The specified model is either not trained, damaged,\
                      or in a wrong path. It should be ../RUN_" + model_abbr + "/args.pkl")
tester = Tester(args=saved_args, model_choice="V", test_batch_size = 32)
tester.test(printout=True, if_testing=True)

print "\nTotal loss = " + str(tester.get_total_loss())
print "\nLoss on each sequence: "
print tester.get_loss_on_each_seq()