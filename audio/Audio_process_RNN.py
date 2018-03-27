#!/usr/bin/python3.6

from __future__ import print_function, division
import os
import json
import sys
import numpy as np
np.set_printoptions(threshold=np.nan)
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf

num_epochs = 1
total_series_length = 25000
truncated_backprop_length = 10
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 1
num_batches = total_series_length//batch_size//truncated_backprop_length

DevFeatures = np.load("dev_features_pca.csv.npy")
DevLabels = np.load("dev_labels_pca.csv.npy")
DevNamedIndices = np.load("dev_named_indices_pca.csv.npy")
TestFeatures = np.load("test_features_pca.csv.npy")
TestLabels = np.load("test_labels_pca.csv.npy")
TestNamedIndices = np.load("test_named_indices_pca.csv.npy")

def generateTrainingData():
	x = DevFeatures[0:26560,:]
	y = DevLabels[0:26560]
	y[0:echo_step] = 0

	x = x.reshape((batch_size, -1))  # The first index changing slowest, subseries as rows
	y = y.reshape((batch_size, -1))

	return (x, y)
	

def generateTestingData():
	x = TestFeatures[0:28680,:]
	y = TestLabels[0:28680]
	y[0:echo_step] = 0

	x = x.reshape((batch_size, -1))  # The first index changing slowest, subseries as rows
	y = y.reshape((batch_size, -1))

	return (x, y)

batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])

init_state = tf.placeholder(tf.float32, [batch_size, state_size])

W2 = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)

# Unstack columns
inputs_series = tf.split(batchX_placeholder, truncated_backprop_length, 1)
labels_series = tf.unstack(batchY_placeholder, axis=1)

# Forward passes
cell = tf.nn.rnn_cell.BasicRNNCell(state_size)
states_series, current_state = tf.contrib.rnn.static_rnn(cell, inputs_series, init_state)

logits_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]

losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = labels) for logits, labels in zip(logits_series,labels_series)]
total_loss = tf.reduce_mean(losses)

train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	loss_list = []

	for epoch_idx in range(num_epochs):
		x,y = generateTrainingData()
		_current_state = np.zeros((batch_size, state_size))

		print("New data, epoch", epoch_idx)

		for batch_idx in range(num_batches):
			start_idx = batch_idx * truncated_backprop_length
			end_idx = start_idx + truncated_backprop_length

			batchX = x[:,start_idx:end_idx]
			batchY = y[:,start_idx:end_idx]

			_total_loss, _train_step, _current_state, _predictions_series = sess.run(
				[total_loss, train_step, current_state, predictions_series],
				feed_dict={
					batchX_placeholder:batchX,
					batchY_placeholder:batchY,
					init_state:_current_state
				})

			loss_list.append(_total_loss)

			if batch_idx%100 == 0:
				print("Step",batch_idx, "Loss", _total_loss)
	
	print("\nTraining completed")
	print("Evaluating the model\n")
	
	loss_list = []
	predict_values_tensor = []
	batchY_tensor = []
	
	for epoch_idx in range(num_epochs):
		x,y = generateTestingData()
		_current_state = np.zeros((batch_size, state_size))

		print("New data, epoch", epoch_idx)
		
		for batch_idx in range(num_batches):
			start_idx = batch_idx * truncated_backprop_length
			end_idx = start_idx + truncated_backprop_length

			batchX = x[:,start_idx:end_idx]
			batchY = y[:,start_idx:end_idx]

			_current_state, _predictions_series = sess.run(
				[current_state, predictions_series],
				feed_dict={
					batchX_placeholder:batchX,
					batchY_placeholder:batchY,
					init_state:_current_state
				})
			
			loss_list.append(_total_loss)

			predict_tensor = tf.concat(_predictions_series[:], 0)
			predict_values_tensor.append(tf.argmax(predict_tensor,1))
			batchY_tensor.append(tf.cast(tf.convert_to_tensor(batchY), tf.int64))
	
	predictions_tensor = tf.concat(predict_values_tensor[:], 0)
	labels_tensor = tf.reshape(tf.concat(batchY_tensor[:], 1), [-1,]) 
	
	accuracy = tf.metrics.accuracy(labels_tensor, predictions_tensor)
	recall = tf.metrics.recall(labels_tensor, predictions_tensor)
	precision = tf.metrics.precision(labels_tensor, predictions_tensor)
	auc = tf.metrics.auc(labels_tensor, predictions_tensor)
	fp = tf.metrics.false_positives(labels_tensor, predictions_tensor)
	tp = tf.metrics.true_positives(labels_tensor, predictions_tensor)
	fn = tf.metrics.false_negatives(labels_tensor, predictions_tensor)
	tn = tf.metrics.true_negatives(labels_tensor, predictions_tensor)
	
	sess.run(tf.initialize_local_variables())
	sess.run(tf.initialize_all_variables())
	
	_accuracy,_recall,_precision,_auc,_fp,_tp,_fn,_tn = sess.run([accuracy,recall,precision,auc,fp,tp,fn,tn])
	
	print("Accuracy is: ",_accuracy[0])
	print("Recall is: ",_recall[0])
	print("Precision is: ",_precision[0])
	print("ROC AUC is: ",_auc[0])
	print("tn, fp, fn, tp are: ",_tn[0], _fp[0], _fn[0], _tp[0])
