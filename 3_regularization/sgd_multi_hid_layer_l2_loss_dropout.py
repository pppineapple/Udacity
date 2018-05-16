#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 22:54:47 2018

@author: pineapple
"""

import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

pickle_file = '../notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)
  
image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

# Adding a hidden layer with 1024 nodes to the nerual network
#sgd with one hidden layer and l2 regularization
batch_size = 128
alpha = 0.01
graph = tf.Graph()
keep_prob = 0.5

nodes = (256,512,1024,512,10)
w1_nodes,w2_nodes,w3_nodes,w4_nodes,w5_nodes=nodes


with graph.as_default():
    
    tf_train_dataset = tf.placeholder(tf.float32, shape=[batch_size, image_size*image_size])
    tf_train_labels = tf.placeholder(tf.float32, shape=[batch_size, num_labels])
    
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_valid_labels = tf.constant(valid_labels)
    
    tf_test_dataset = tf.constant(test_dataset)
    tf_test_labels = tf.constant(test_labels)
    
    global_step = tf.Variable(0)
    
    w1 = tf.Variable(tf.truncated_normal([image_size*image_size, w1_nodes],
                                         stddev=np.sqrt(2.0 / (image_size * image_size))))
    b1 = tf.Variable(tf.zeros([w1_nodes]))
    
    w2 = tf.Variable(tf.truncated_normal([w1_nodes, w2_nodes],
                                         stddev=np.sqrt(2.0 / w1_nodes)))    
    b2 = tf.Variable(tf.zeros([w2_nodes]))
    
    w3 = tf.Variable(tf.truncated_normal([w2_nodes, w3_nodes],
                                         stddev=np.sqrt(2.0 / w2_nodes)))
    b3 = tf.Variable(tf.zeros([w3_nodes]))
    
    w4 = tf.Variable(tf.truncated_normal([w3_nodes, w4_nodes],
                                         stddev=np.sqrt(2.0 / w3_nodes)))
    b4 = tf.Variable(tf.zeros([w4_nodes]))
    
    w5 = tf.Variable(tf.truncated_normal([w4_nodes, w5_nodes],
                                         stddev=np.sqrt(2.0 / w4_nodes)))
    b5 = tf.Variable(tf.zeros([w5_nodes]))
    
    #keep_prob = tf.Variable(tf.truncated_normal([1])) >>> Error
    
    w1_output = tf.matmul(tf_train_dataset, w1) + b1
    w1_output = tf.nn.relu(w1_output)
#    w1_output = tf.nn.dropout(tf.nn.relu(w1_output), keep_prob)
#    w1_output = tf.nn.dropout(w1_output, keep_prob)
    w2_output = tf.matmul(w1_output, w2) + b2
    w2_output = tf.nn.relu(w2_output)
#    w2_output = tf.nn.dropout(tf.nn.relu(w2_output), keep_prob)
#    w2_output = tf.nn.dropout(w2_output, keep_prob)    
    w3_output = tf.matmul(w2_output, w3) + b3
    w3_output = tf.nn.relu(w3_output)
#    w3_output = tf.nn.dropout(tf.nn.relu(w3_output), keep_prob)
#    w3_output = tf.nn.dropout(w3_output, keep_prob)
    w4_output = tf.matmul(w3_output, w4) + b4
    w4_output = tf.nn.relu(w4_output)
#    w4_output = tf.nn.dropout(tf.nn.relu(w4_output), keep_prob)
#    w4_output = tf.nn.dropout(w4_output, keep_prob)
    w5_output = tf.matmul(w4_output, w5) + b5
    w5_output = tf.nn.relu(w5_output)
#    w5_output = tf.nn.dropout(tf.nn.relu(w5_output), keep_prob)
#    w5_output = tf.nn.dropout(w5_output, keep_prob)
    w5_output = tf.nn.relu(w5_output)
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits = w5_output))
    loss = loss + alpha * (tf.nn.l2_loss(w1) + tf.nn.l2_loss(b1)
                         + tf.nn.l2_loss(w2) + tf.nn.l2_loss(b2)
                         + tf.nn.l2_loss(w3) + tf.nn.l2_loss(b3)
                         + tf.nn.l2_loss(w4) + tf.nn.l2_loss(b4)
                         + tf.nn.l2_loss(w5) + tf.nn.l2_loss(b5))
    
    learning_rate = tf.train.exponential_decay(0.1, global_step, 100, 0.9, staircase=True)
    #learning_rate：0.1；staircase=True;则每100轮训练后要乘以0.96.                 
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    
    train_prediction = tf.nn.softmax(w5_output)
    
    valid_w1_output = tf.matmul(tf_valid_dataset, w1) + b1
    valid_w1_output = tf.nn.relu(valid_w1_output)
#    valid_w1_output = tf.nn.dropout(tf.nn.relu(valid_w1_output), keep_prob)
#    valid_w1_output = tf.nn.dropout(valid_w1_output, keep_prob)    
    valid_w2_output = tf.matmul(valid_w1_output, w2) + b2
    valid_w2_output = tf.nn.relu(valid_w2_output)
#    valid_w2_output = tf.nn.dropout(tf.nn.relu(valid_w2_output), keep_prob)
#    valid_w2_output = tf.nn.dropout(valid_w2_output, keep_prob)    
    valid_w3_output = tf.matmul(valid_w2_output, w3) + b3
    valid_w3_output = tf.nn.relu(valid_w3_output)
#    valid_w3_output = tf.nn.dropout(tf.nn.relu(valid_w3_output), keep_prob)    
#    valid_w3_output = tf.nn.dropout(valid_w3_output, keep_prob)
    valid_w4_output = tf.matmul(valid_w3_output, w4) + b4
    valid_w4_output = tf.nn.relu(valid_w4_output)
#    valid_w4_output = tf.nn.dropout(tf.nn.relu(valid_w4_output), keep_prob)    
#    valid_w4_output = tf.nn.dropout(valid_w4_output, keep_prob)
    valid_w5_output = tf.matmul(valid_w4_output, w5) + b5
    valid_w5_output = tf.nn.relu(valid_w5_output)
#    valid_w5_output = tf.nn.dropout(tf.nn.relu(valid_w5_output), keep_prob)
#    valid_w5_output = tf.nn.dropout(valid_w5_output, keep_prob)
    valid_prediction = tf.nn.softmax(valid_w5_output)

    test_w1_output = tf.matmul(tf_test_dataset, w1) + b1
    test_w1_output = tf.nn.relu(test_w1_output)
#    test_w1_output = tf.nn.dropout(tf.nn.relu(test_w1_output), keep_prob)
#    test_w1_output = tf.nn.dropout(test_w1_output, keep_prob)    
    test_w2_output = tf.matmul(test_w1_output, w2) + b2
    test_w2_output = tf.nn.relu(test_w2_output)
#    test_w2_output = tf.nn.dropout(tf.nn.relu(test_w2_output), keep_prob)
#    test_w2_output = tf.nn.dropout(test_w2_output, keep_prob)    
    test_w3_output = tf.matmul(test_w2_output, w3) + b3
    test_w3_output = tf.nn.relu(test_w3_output)
#    test_w3_output = tf.nn.dropout(tf.nn.relu(test_w3_output), keep_prob)    
#    test_w3_output = tf.nn.dropout(test_w3_output, keep_prob)
    test_w4_output = tf.matmul(test_w3_output, w4) + b4
    test_w4_output = tf.nn.relu(test_w4_output)
#    test_w4_output = tf.nn.dropout(tf.nn.relu(test_w4_output), keep_prob)    
#    test_w4_output = tf.nn.dropout(test_w4_output, keep_prob)
    test_w5_output = tf.matmul(test_w4_output, w5) + b5
    test_w5_output = tf.nn.relu(test_w5_output)
#    test_w5_output = tf.nn.dropout(tf.nn.relu(test_w5_output), keep_prob)
#    test_w5_output = tf.nn.dropout(test_w5_output, keep_prob)
    test_prediction = tf.nn.softmax(test_w5_output)
   

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])
  
num_steps = 3001

with tf.Session(graph=graph) as session:
    
    tf.global_variables_initializer().run()
    print 'Initialized'
    
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        
        feed_dict = {tf_train_dataset : batch_data,tf_train_labels : batch_labels}
        #_, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        
        _, l, prediction = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        
        if (step % 500 == 0):
            print 'Minibatch loss at step %s: %s' % (step, l)
            print 'Minibatch accuracy:  %.1f%%' % (accuracy(prediction, batch_labels))
            print 'Validation accuracy: %.1f%%' % (accuracy(valid_prediction.eval(), valid_labels))
    print 'Test accuracy: %.1f%%' % (accuracy(test_prediction.eval(), test_labels))