import tensorflow as tf
import sys
import model
import time
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

seed = 0
tf.set_random_seed( seed )
 
start_time = time.time()

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

######################################################
# mnist为tensorflow中的Datasets型
 
# mnist.train.images        (55000, 784)    范围[0, 1]
# mnist.train.labels        (55000, 10)
 
# mnist.validation.images   (5000, 784)
# mnist.validation.labels   (5000, 10)
 
# mnist.test.images         (10000, 784)
# mnist.test.labels         (10000, 10)
######################################################

x = tf.placeholder("float", [None, 784])
#W = tf.Variable(tf.zeros([784,10]))
#b = tf.Variable(tf.zeros([10]))
 
#y = tf.nn.softmax(tf.matmul(x,W) + b)
y_hat = model.create_model(x, 10)

y = tf.placeholder("float", [None,10])
loss = model.create_loss(y, y_hat)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_hat,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step = tf.train.AdamOptimizer( 1e-4 ).minimize( loss )
 
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

BATCH_SIZE = 500
NUM_SAMPLES = len( mnist.train.images )
NUM_BATCHES = NUM_SAMPLES // BATCH_SIZE

for epoch in range(300):
    
    t0 = time.time()
    cur_loss = 0

    for i in range( NUM_BATCHES ):

        batch_x, batch_y = mnist.train.next_batch( BATCH_SIZE )
        dummy, cur_batch_loss = sess.run([train_step, loss],
                                         feed_dict={x: batch_x, y: batch_y})
        cur_loss += cur_batch_loss

    print( 'epoch = %d\tcur_loss = %f\tin %.2f s' % ( epoch, cur_loss / NUM_BATCHES,
                                                      time.time() - t0 ) )
    

[train_loss, train_accuracy] = sess.run( [loss, accuracy], feed_dict={ x: mnist.train.images, y: mnist.train.labels} )
[valid_loss, valid_accuracy] = sess.run( [loss, accuracy], feed_dict={ x: mnist.validation.images, y: mnist.validation.labels} )
[test_loss, test_accuracy] = sess.run( [loss, accuracy], feed_dict={ x: mnist.test.images, y: mnist.test.labels} )
 
print( '\nTrain Set: loss = %f, accuracy = %f' % ( train_loss, train_accuracy ) )
print( 'Valid Set: loss = %f, accuracy = %f' % ( valid_loss, valid_accuracy ) )
print( 'Test Set: loss = %f, accuracy = %f' % ( test_loss, test_accuracy ) )
 
 
 
print( '\nDone in %.2f min' % ( ( time.time() - start_time ) / 60 ) )
[test_loss, test_accuracy] = sess.run( [loss, accuracy], feed_dict={x: mnist.test.images, y: mnist.test.labels})
print( 'Test Set: loss = %f, accuracy = %f' % ( test_loss, test_accuracy ) )


#--------------------------------------------
#After ratation
test_imgs = np.load("test_imgs_rotate.npy")
[test_loss, test_accuracy] = sess.run( [loss, accuracy], feed_dict={x: test_imgs, y: mnist.test.labels})
print("After rotation, Test set: loss = %f, accuracy = %f" %( test_loss, test_accuracy))

#--------------------------------------------
#After trans
test_imgs = np.load("test_imgs_trans.npy")
[test_loss, test_accuracy] = sess.run( [loss, accuracy], feed_dict={x: test_imgs, y: mnist.test.labels})
print("After trans, Test set: loss = %f, accuracy = %f" %( test_loss, test_accuracy))
