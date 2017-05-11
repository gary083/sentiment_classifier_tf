import collections
import os
import tensorflow as tf 
import numpy as np
import model
import argparse

embedding_dim = 128
max_length = 30
unit_size = 128
batch_size = 1024
load_model = False

def parse():
	parser = argparse.ArgumentParser()
	parser.add_argument('-train', action='store_true', help='train...')
	parser.add_argument('-test', action='store_true', help='test...')
	parser.add_argument('-embedding_dim' , default=128, help='embedding_dim...')
	parser.add_argument('-max_length' , default=30, help='max_length...')
	parser.add_argument('-unit_size' , default=128, help='unit_size...')
	parser.add_argument('-batch_size' , default=1024, help='batch_size...')
	parser.add_argument('-load_model' , default=False, help='load_model...')
	parser.add_argument('-model_dir' , default='save', help='model_dir...')
	parser.add_argument('-dict_file' , default='dictionary.txt', help='dict_file...')
	args = parser.parse_args()
	return args

def run(args):
	sess = tf.Session()
	classifier =  model.sentiment_classifier(sess, args)
	if arg.train:
		classifier.train()
	elif arg.test:
		classifier.test()

if __name__ == '__main__':
	arg = parse()
	run(arg)
