from tensorflow.contrib import slim
import tensorflow as tf
from handler_data import get_hed,get_test_data
import numpy as np
from matplotlib import pyplot as plt
import  os
checkpoints_dir ='pig'
import cv2

def vgg_arg_scope(weight_decay=0.0005):
  """Defines the VGG arg scope.

  Args:
    weight_decay: The l2 regularization coefficient.

  Returns:
    An arg_scope.
  """
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer()):
    with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
      return arg_sc

image = tf.placeholder(dtype=tf.float32,shape=(1,512,512,3))


def create_model(inputs):
    with slim.arg_scope(vgg_arg_scope()):
        with tf.variable_scope('vgg_16', 'vgg_16', [inputs]) as sc:
            end_points_collection = sc.name + '_end_points'
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with slim.arg_scope([slim.conv2d,slim.max_pool2d],
                                outputs_collections=end_points_collection):
                with slim.arg_scope([slim.conv2d,slim.conv2d_transpose],trainable=False):
                    cv1 = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                    p1 = slim.max_pool2d(cv1, [2, 2], scope='pool1')
                    cv2 = slim.repeat(p1, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                    p2 = slim.max_pool2d(cv2, [2, 2], scope='pool2')
                    cv3 = slim.repeat(p2, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                    p3 = slim.max_pool2d(cv3, [2, 2], scope='pool3')
                    cv4 = slim.repeat(p3, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                    p4 = slim.max_pool2d(cv4, [2, 2], scope='pool4')
                    cv5 = slim.repeat(p4, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                    svs = slim.get_variables_to_restore()


                    cv1_1 = slim.conv2d(cv1,num_outputs=1,kernel_size=1,stride=1,activation_fn=tf.nn.tanh)



                    cv2_1 = slim.conv2d(cv2,num_outputs=1,kernel_size=1,stride=1,activation_fn=None)
                    dev2 = slim.conv2d_transpose(cv2_1,num_outputs=1,kernel_size=4,stride=2,activation_fn=tf.nn.tanh)


                    cv3_1 = slim.conv2d(cv3, num_outputs=1, kernel_size=1, stride=1,activation_fn=None)
                    dev3 = slim.conv2d_transpose(cv3_1, num_outputs=1, kernel_size=8, stride=4,activation_fn=tf.nn.tanh)


                    cv4_1 = slim.conv2d(cv4, num_outputs=1, kernel_size=1, stride=1,activation_fn=None)
                    dev4 = slim.conv2d_transpose(cv4_1, num_outputs=1, kernel_size=16, stride=8,activation_fn=tf.nn.tanh)


                    cv5_1 = slim.conv2d(cv5, num_outputs=1, kernel_size=1, stride=1,activation_fn=None)
                    dev5 = slim.conv2d_transpose(cv5_1, num_outputs=1, kernel_size=32, stride=16,activation_fn=tf.nn.tanh)


                    ct = tf.concat([cv1_1,dev2,dev3,dev4,dev5],3)
                    final_cv = slim.conv2d(ct,num_outputs=1,kernel_size=1,stride=1,weights_initializer=tf.constant_initializer(0.2),activation_fn=tf.nn.tanh)




                    return final_cv

img = create_model(image)
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(checkpoints_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    for pp in os.listdir('/home/dsl/PycharmProjects/gen_pig/org'):
        fin = sess.run(img,feed_dict={image:get_test_data('/home/dsl/PycharmProjects/gen_pig/org/'+pp)})
        ne = np.zeros(shape=(512,512,3))
        ne[:,:,0] = fin[0,:,:,0]*255
        ne[:, :, 1] = fin[0, :, :, 0]*255
        ne[:, :, 2] = fin[0, :, :, 0]*255
        d0 = np.where(ne>0)
        ne[d0] = 255

        cv2.imwrite('gen/'+pp,ne)
        #plt.imshow(fin[0,:,:,0],cmap='gray')
        #plt.show()