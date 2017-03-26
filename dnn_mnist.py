# -*- coding: utf-8 -*-

"""
This program builds a deep neural network model to classify MNIST data
The network consists of three hidden layers of size 500-1500-200
Training is done for 10 epochs with a batch size of 150

Author: Ankit Bansal
Date: 03-25-2017
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist_data/", one_hot = True)

input_dim = 28*28

nodes_hidden_layers_1 = 500
nodes_hidden_layers_2 = 1500
nodes_hidden_layers_3 = 200

n_classes = 10

batch_size = 150

# Input data flatten to 1D
x = tf.placeholder('float',[None, input_dim])
y = tf.placeholder('float')

def neural_network_model(data):
    
    # Creates a tf variable for weights of hidden layer 1 of shape 28*28 X 500
    hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([input_dim, nodes_hidden_layers_1])),
                      'biases':tf.Variable(tf.random_normal([nodes_hidden_layers_1]))}
    
    hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([nodes_hidden_layers_1, nodes_hidden_layers_2])),
                      'biases':tf.Variable(tf.random_normal([nodes_hidden_layers_2]))}
        
    hidden_layer_3 = {'weights': tf.Variable(tf.random_normal([nodes_hidden_layers_2, nodes_hidden_layers_3])),
                      'biases':tf.Variable(tf.random_normal([nodes_hidden_layers_3]))}
            
    output_layer = {'weights': tf.Variable(tf.random_normal([nodes_hidden_layers_3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes]))}
    
    # input*weights + biases
    output_hidden_layer_1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), hidden_layer_1['biases'])
    
    # Activation function 
    activated_output_l1 = tf.nn.relu(output_hidden_layer_1)
    
    output_hidden_layer_2 = tf.add(tf.matmul(activated_output_l1, hidden_layer_2['weights']), hidden_layer_2['biases'])
    activated_output_l2 = tf.nn.relu(output_hidden_layer_2)

    output_hidden_layer_3 = tf.add(tf.matmul(activated_output_l2, hidden_layer_3['weights']), hidden_layer_3['biases'])
    activated_output_l3 = tf.nn.relu(output_hidden_layer_3)   
    
    output = tf.add(tf.matmul(activated_output_l3, output_layer['weights']), output_layer['biases'])
     
    return output

def train_network(x,y):
    
    prediction = neural_network_model(x)
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    
    no_epochs = 10
    
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        
        # Train
        for epoch in range(no_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                _, l = sess.run([optimizer,loss], feed_dict = {x: batch_x, y: batch_y})\
                
                epoch_loss += l
            print('############################')    
            print('Epoch ', epoch + 1, ' out of ', no_epochs)
            print('Loss: ', epoch_loss) 
            
       
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        
        # Compute Accuracy 
        print('Accuracy: ', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        
    
train_network(x, y) 

     
        
        
        
            