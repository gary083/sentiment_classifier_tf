import collections
import os
import tensorflow as tf 
import numpy as np
import model

embedding_dim = 128
max_length = 30
unit_size = 128
batch_size = 1024
load_model = False

def run(mode):
	sess = tf.Session()
	classifier =  model.sentiment_classifier(sess, embedding_dim, max_length, unit_size, batch_size, load_model)
	if mode == 'train':
		classifier.train()
	elif mode == 'test':
		classifier.test()


if __name__ == '__main__':
	#mode = 'train'
	mode = 'test'
	run(mode)
