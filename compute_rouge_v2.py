"""
ROUGE calculator implemented with Python 2.7.12 and Pyrouge 5.5.1
- - - - -
CS224N Project at Stanford Univeristy
Project mentor: Prof. Chris Manning

Author: W.Z.Zhou
Date: March 5, 2017
- - - - -
Calculate ROUGE.
"""
from pythonrouge.pythonrouge import Pythonrouge
import sys

ROUGE_path = '../pythonrouge/pythonrouge/RELEASE-1.5.5/ROUGE-1.5.5.pl'
data_path = '../pythonrouge/pythonrouge/RELEASE-1.5.5/data' #data folder in RELEASE-1.5.5

summary_path = "../actual_words/canonical_seq100_feed_input.txt"
label_path = "../test_true_label.txt"

sum_list = []
label_list = []

with open(summary_path, "r") as fs:
	for line in fs:
		# print type(line)
		line.rstrip("\n")
		# print line
		temp = []
		temp.append(line)
		sum_list.append(temp)

print "there are ", len(sum_list), " sentences in prediction summary."
# print sum_list

with open(label_path, "r") as fin:
	for line in fin:
		line.rstrip("\n")
		# print line
		temp1 = []
		temp1.append(line)
		temp2 = []
		temp2.append(temp1)
		label_list.append(temp2)

print "there are ", len(label_list), " sentences in label summary."
# print label_list

# initialize setting of ROUGE, eval ROUGE-1, 2, SU4, L
rouge = Pythonrouge(n_gram=2, ROUGE_SU4=True, ROUGE_L=True, stemming=True, stopwords=True, word_level=True, length_limit=True, length=50, use_cf=False, cf=95, scoring_formula="average", resampling=True, samples=1000, favor=True, p=0.5)

# system summary & reference summary
# summary = [[" Tokyo is the one of the biggest city in the world."],["I like dogs."]]
# reference = [ [["The capital of Japan, Tokyo, is the center of Japanese economy."],[" Tokyo is the biggest city in the world."]],
#              [["I like dogs."]]
#             ] 

# If you evaluate ROUGE by sentence list as above, set files=False
setting_file = rouge.setting(files=False, summary=sum_list, reference=label_list)

# print "setting_file is: ", setting_file
# If you need only recall of ROUGE metrics, set recall_only=True
result = rouge.eval_rouge(setting_file, recall_only=True, ROUGE_path=ROUGE_path, data_path=data_path)
print(result)

