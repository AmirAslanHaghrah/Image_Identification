# <Gun_NotGun_Image_Identification>
# <Simple_Gun_NotGun_Classification_NN_Tensorflow>
# Copyright © <2018> <AmirAslan Haghrah>

# Permission is hereby granted, free of charge, to any person obtaining a copy of 
# this software and associated documentation files (the "Software"), to deal in 
# the Software without restriction, including without limitation the rights 
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
# of the Software, and to permit persons to whom the Software is furnished to do 
# so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import numpy as np
import tensorflow as tf

Gun_train_data = np.loadtxt('packedData/Gun_Images_train_data_2D.txt', dtype=int)              # Import Train Gun data
NotGun_train_data = np.loadtxt('packedData/NotGun_Images_train_data_2D.txt', dtype=int)        # Import Train NotGun data

# Reshaped loaded Train data to be compatible with tensorflow
Gun_num = np.shape(Gun_train_data)[0]
NotGun_num = np.shape(NotGun_train_data)[0]
input_data = np.r_[Gun_train_data, NotGun_train_data]
output_data = np.c_[np.r_[np.ones(shape=(Gun_num, 1)), np.zeros(shape=(Gun_num, 1))], np.r_[np.zeros(shape=(NotGun_num, 1)), np.ones(shape=(NotGun_num, 1))]]


# Create the model
x = tf.placeholder(tf.float32, [None, 73728])
W = tf.Variable(tf.zeros([73728, 2]))
b = tf.Variable(tf.zeros([2]))
y = tf.matmul(x, W) + b

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 2])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


# Train
for _ in range(100):
    sess.run(train_step, feed_dict={x: input_data, y_: output_data})

 
############################### Test NN Accuracy ####################################
Gun_test_data = np.loadtxt('packedData/Gun_Images_test_data_2D.txt', dtype=int)                # Import Test Gun data
NotGun_test_data = np.loadtxt('packedData/NotGun_Images_test_data_2D.txt', dtype=int)          # Import Test NotGun data

# Reshaped loaded Train data to be compatible with tensorflow
Gun_num = np.shape(Gun_test_data)[0]
NotGun_num = np.shape(NotGun_test_data)[0]
input_data = np.r_[Gun_test_data, NotGun_test_data]
output_data = np.c_[np.r_[np.ones(shape=(Gun_num, 1)), np.zeros(shape=(Gun_num, 1))], np.r_[np.zeros(shape=(NotGun_num, 1)), np.ones(shape=(NotGun_num, 1))]]

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: input_data, y_: output_data}))
####################################################################################
