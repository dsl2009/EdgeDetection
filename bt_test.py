from nets import resnet_v2
import tensorflow as tf
from tensorflow.contrib import slim
from matplotlib import pyplot as plt
import numpy as np
y = tf.constant([1.0,1,0])
y_ = tf.constant([1.0,1,0])
image_string = open('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/edge_detect/tea/org/27.jpg', 'r').read()
image = tf.image.decode_jpeg(image_string, channels=3)




input_image = tf.image.convert_image_dtype(image, dtype=tf.float32)


loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=y_)

sg = tf.nn.sigmoid(y_)
sg1 = tf.nn.sigmoid(y)
sg3 = y*-tf.log(sg)+(1-y)*-tf.log(1-sg)
d = tf.reduce_mean(sg3)


with tf.Session() as sess:
    print sess.run(image)
    print np.where(sess.run(input_image)==0)

