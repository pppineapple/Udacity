#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 20:43:11 2018

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

#sgd with one layer and l2 regularization
batch_size = 128 
alpha = 0.01

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels]))
  biases = tf.Variable(tf.zeros([num_labels]))

  
  # Training computation.
  logits = tf.matmul(tf_train_dataset, weights) + biases
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
  loss = loss + alpha * tf.nn.l2_loss(weights)
  
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
  test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
  
num_steps = 3001

with tf.Session(graph=graph) as session:
  #tf.train.write_graph(sess.graph,'sgd_graph','sgd_graph.pb') 
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
  
#sgd with one hidden layer and l2 regularization
batch_size = 128
alpha = 0.01
graph = tf.Graph()

with graph.as_default():
    
    tf_train_dataset = tf.placeholder(tf.float32, shape=[batch_size, image_size*image_size])
    tf_train_labels = tf.placeholder(tf.float32, shape=[batch_size, num_labels])
    
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_valid_labels = tf.constant(valid_labels)
    
    tf_test_dataset = tf.constant(test_dataset)
    tf_test_labels = tf.constant(test_labels)
    
    hidden_layer_weights = tf.Variable(tf.truncated_normal([image_size*image_size, 1024]))
    hidden_layer_biases = tf.Variable(tf.zeros([1024]))
    
    output_layer_weights = tf.Variable(tf.truncated_normal([1024, num_labels]))
    output_layer_biases = tf.Variable(tf.zeros([num_labels]))
    
    hidden_layer_output = tf.matmul(tf_train_dataset, hidden_layer_weights) + hidden_layer_biases
    hidden_layer_output = tf.nn.relu(hidden_layer_output)
    output_layer_logit = tf.matmul(hidden_layer_output, output_layer_weights) + output_layer_biases

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels,logits=output_layer_logit))
    loss = loss + alpha * (tf.nn.l2_loss(hidden_layer_weights) + tf.nn.l2_loss(hidden_layer_biases) 
                + tf.nn.l2_loss(output_layer_weights) + tf.nn.l2_loss(output_layer_biases))
            
    
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    
    train_prediction = tf.nn.softmax(output_layer_logit)
    
    valid_hidden_layer_output = tf.matmul(tf_valid_dataset, hidden_layer_weights)+hidden_layer_biases
    valid_hidden_layer_output = tf.nn.relu(valid_hidden_layer_output)
    valid_output_layer_logit = tf.matmul(valid_hidden_layer_output, output_layer_weights)+output_layer_biases
    valid_prediction = tf.nn.softmax(valid_output_layer_logit)
    
    test_hidden_layer_output = tf.matmul(tf_test_dataset, hidden_layer_weights)+hidden_layer_biases
    test_hidden_layer_output = tf.nn.relu(test_hidden_layer_output)
    test_output_layer_logit = tf.matmul(test_hidden_layer_output, output_layer_weights)+output_layer_biases
    test_prediction = tf.nn.softmax(test_output_layer_logit)

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


# sgd with one layer and few train dataset 
batch_size = 128
few_batch_size = 5 * batch_size
alpha = 0.01

few_train_dataset = train_dataset[:few_batch_size,:]
few_train_labels = train_labels[:few_batch_size,:]

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels]))
  biases = tf.Variable(tf.zeros([num_labels]))
  
  # Training computation.
  logits = tf.matmul(tf_train_dataset, weights) + biases
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
  loss = loss + alpha * tf.nn.l2_loss(weights)
  
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
  test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
  
num_steps = 3001

with tf.Session(graph=graph) as session:
  #tf.train.write_graph(sess.graph,'sgd_graph','sgd_graph.pb') 
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (few_train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = few_train_dataset[offset:(offset + batch_size), :]
    batch_labels = few_train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
  
#sgd with one hidden layer and few train dataset

batch_size = 128
graph = tf.Graph()

few_batch_size = 5 * batch_size
alpha = 0.01

few_train_dataset = train_dataset[:few_batch_size,:]
few_train_labels = train_labels[:few_batch_size,:]

with graph.as_default():
    
    tf_train_dataset = tf.placeholder(tf.float32, shape=[batch_size, image_size*image_size])
    tf_train_labels = tf.placeholder(tf.float32, shape=[batch_size, num_labels])
    
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_valid_labels = tf.constant(valid_labels)
    
    tf_test_dataset = tf.constant(test_dataset)
    tf_test_labels = tf.constant(test_labels)
    
    hidden_layer_weights = tf.Variable(tf.truncated_normal([image_size*image_size, 1024]))
    hidden_layer_biases = tf.Variable(tf.zeros([1024]))
    
    output_layer_weights = tf.Variable(tf.truncated_normal([1024, num_labels]))
    output_layer_biases = tf.Variable(tf.zeros([num_labels]))
    
    hidden_layer_output = tf.matmul(tf_train_dataset, hidden_layer_weights) + hidden_layer_biases
    hidden_layer_output = tf.nn.relu(hidden_layer_output)
    output_layer_logit = tf.matmul(hidden_layer_output, output_layer_weights) + output_layer_biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels,logits=output_layer_logit))
    loss = loss + alpha * (tf.nn.l2_loss(hidden_layer_weights) + tf.nn.l2_loss(hidden_layer_biases) 
                + tf.nn.l2_loss(output_layer_weights) + tf.nn.l2_loss(output_layer_biases))
    
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    
    train_prediction = tf.nn.softmax(output_layer_logit)
    
    valid_hidden_layer_output = tf.matmul(tf_valid_dataset, hidden_layer_weights)+hidden_layer_biases
    valid_hidden_layer_output = tf.nn.relu(valid_hidden_layer_output)
    valid_output_layer_logit = tf.matmul(valid_hidden_layer_output, output_layer_weights)+output_layer_biases
    valid_prediction = tf.nn.softmax(valid_output_layer_logit)
    
    test_hidden_layer_output = tf.matmul(tf_test_dataset, hidden_layer_weights)+hidden_layer_biases
    test_hidden_layer_output = tf.nn.relu(test_hidden_layer_output)
    test_output_layer_logit = tf.matmul(test_hidden_layer_output, output_layer_weights)+output_layer_biases
    test_prediction = tf.nn.softmax(test_output_layer_logit)

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])
  
    
num_steps = 3001

with tf.Session(graph=graph) as session:
    
    tf.global_variables_initializer().run()
    print 'Initialized'
    
    for step in range(num_steps):
        offset = (step * batch_size) % (few_train_labels.shape[0] - batch_size)
        
        batch_data = few_train_dataset[offset:(offset + batch_size), :]
        batch_labels = few_train_labels[offset:(offset + batch_size), :]
        
        feed_dict = {tf_train_dataset : batch_data,tf_train_labels : batch_labels}
        #_, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        
        _, l, prediction = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        
        if (step % 500 == 0):
            print 'Minibatch loss at step %s: %s' % (step, l)
            print 'Minibatch accuracy:  %.1f%%' % (accuracy(prediction, batch_labels))
            print 'Validation accuracy: %.1f%%' % (accuracy(valid_prediction.eval(), valid_labels))
    print 'Test accuracy: %.1f%%' % (accuracy(test_prediction.eval(), test_labels))


# sgd with hidden layer and l2 regularization and dropput
batch_size = 128
alpha = 0.01
graph = tf.Graph()
keep_prob = 0.5

with graph.as_default():
    
    tf_train_dataset = tf.placeholder(tf.float32, shape=[batch_size, image_size*image_size])
    tf_train_labels = tf.placeholder(tf.float32, shape=[batch_size, num_labels])
    
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_valid_labels = tf.constant(valid_labels)
    
    tf_test_dataset = tf.constant(test_dataset)
    tf_test_labels = tf.constant(test_labels)
    
    hidden_layer_weights = tf.Variable(tf.truncated_normal([image_size*image_size, 1024]))
    hidden_layer_biases = tf.Variable(tf.zeros([1024]))
    
    output_layer_weights = tf.Variable(tf.truncated_normal([1024, num_labels]))
    output_layer_biases = tf.Variable(tf.zeros([num_labels]))
    
    #keep_prob = tf.Variable(tf.truncated_normal([1])) >>> Error
    
    hidden_layer_output = tf.matmul(tf_train_dataset, hidden_layer_weights) + hidden_layer_biases
    hidden_layer_output = tf.nn.relu(hidden_layer_output)
    hidden_layer_output = tf.nn.dropout(hidden_layer_output, keep_prob)
    
    output_layer_logit = tf.matmul(hidden_layer_output, output_layer_weights) + output_layer_biases

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels,logits=output_layer_logit))
    loss = loss + alpha * (tf.nn.l2_loss(hidden_layer_weights) + tf.nn.l2_loss(hidden_layer_biases) 
                + tf.nn.l2_loss(output_layer_weights) + tf.nn.l2_loss(output_layer_biases))
            
    
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    
    train_prediction = tf.nn.softmax(output_layer_logit)
    
    valid_hidden_layer_output = tf.matmul(tf_valid_dataset, hidden_layer_weights)+hidden_layer_biases
    valid_hidden_layer_output = tf.nn.relu(valid_hidden_layer_output)
    valid_output_layer_logit = tf.matmul(valid_hidden_layer_output, output_layer_weights)+output_layer_biases
    valid_prediction = tf.nn.softmax(valid_output_layer_logit)
    
    test_hidden_layer_output = tf.matmul(tf_test_dataset, hidden_layer_weights)+hidden_layer_biases
    test_hidden_layer_output = tf.nn.relu(test_hidden_layer_output)
    test_output_layer_logit = tf.matmul(test_hidden_layer_output, output_layer_weights)+output_layer_biases
    test_prediction = tf.nn.softmax(test_output_layer_logit)

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