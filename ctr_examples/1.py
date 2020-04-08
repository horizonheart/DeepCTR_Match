import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Embedding, Input, Flatten
sess = tf.InteractiveSession()

embedding = Embedding(6,4,mask_zero=True)
input_ids = tf.placeholder(dtype=tf.int32, shape=[None])
input_embedding = embedding(input_ids)

sess.run(tf.global_variables_initializer())
print sess.run(embedding.weights)
print "====="*100
print sess.run(input_embedding, feed_dict={input_ids: [4, 1, 2, 4, 0, 0, 0, 0]})
print sess.run(embedding.compute_mask(input_ids), feed_dict={input_ids: [4, 1, 2, 4, 0, 0, 0, 0]})