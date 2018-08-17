from tensorflow.contrib import slim
import tensorflow as tf
from handler_data import get_hed
import numpy as np
from matplotlib import pyplot as plt


def sigmoid_cross_entropy_balanced(logits, labels, name='cross_entropy_loss'):
    """
    Implements Equation [2] in https://arxiv.org/pdf/1504.06375.pdf
    Compute edge pixels for each training sample and set as pos_weights to
    tf.nn.weighted_cross_entropy_with_logits
    """
    y = tf.cast(label, tf.float32)

    count_neg = tf.reduce_sum(1. - y)
    count_pos = tf.reduce_sum(y)

    # Equation [2]
    beta = count_neg / (count_neg + count_pos)

    # Equation [2] divide by 1 - beta
    pos_weight = beta / (1 - beta)

    cost = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=y, pos_weight=pos_weight)

    # Multiply by 1 - beta
    cost = tf.reduce_mean(cost * (1 - beta))

    # check if image has no edge pixels return 0 else return complete error function
    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost, name=name)


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
label = tf.placeholder(dtype=tf.float32,shape=(1,512,512,1))

def create_model(inputs, labels):
    with slim.arg_scope(vgg_arg_scope()):
        with tf.variable_scope('vgg_16', 'vgg_16', [inputs]) as sc:
            end_points_collection = sc.name + '_end_points'
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with slim.arg_scope([slim.conv2d,slim.max_pool2d],
                                outputs_collections=end_points_collection):
                cv1 = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                p1 = slim.max_pool2d(cv1, [2, 2], scope='pool1')
                cv2 = slim.repeat(p1, 2, slim.conv2d, 128, [3, 3], scope='conv2',rate=2)
                p2 = slim.max_pool2d(cv2, [2, 2], scope='pool2')
                cv3 = slim.repeat(p2, 3, slim.conv2d, 256, [3, 3], scope='conv3',rate=2)
                p3 = slim.max_pool2d(cv3, [2, 2], scope='pool3')
                cv4 = slim.repeat(p3, 3, slim.conv2d, 512, [3, 3], scope='conv4',rate=2)
                p4 = slim.max_pool2d(cv4, [2, 2], scope='pool4')
                cv5 = slim.repeat(p4, 3, slim.conv2d, 512, [3, 3], scope='conv5',rate=2)
                svs = slim.get_variables_to_restore()


                cv1_1 = slim.conv2d(cv1,num_outputs=1,kernel_size=1,stride=1,activation_fn=tf.nn.sigmoid)
                loss1 = sigmoid_cross_entropy_balanced(labels=labels,logits=cv1_1)


                cv2_1 = slim.conv2d(cv2,num_outputs=1,kernel_size=1,stride=1,activation_fn=None)
                dev2 = slim.conv2d_transpose(cv2_1,num_outputs=1,kernel_size=4,stride=2,activation_fn=tf.nn.sigmoid)
                loss2 = sigmoid_cross_entropy_balanced(labels=labels, logits=dev2)

                cv3_1 = slim.conv2d(cv3, num_outputs=1, kernel_size=1, stride=1,activation_fn=None)
                dev3 = slim.conv2d_transpose(cv3_1, num_outputs=1, kernel_size=8, stride=4,activation_fn=tf.nn.sigmoid)
                loss3 = sigmoid_cross_entropy_balanced(labels=labels, logits=dev3)

                cv4_1 = slim.conv2d(cv4, num_outputs=1, kernel_size=1, stride=1,activation_fn=None)
                dev4 = slim.conv2d_transpose(cv4_1, num_outputs=1, kernel_size=16, stride=8,activation_fn=tf.nn.sigmoid)

                loss4 = sigmoid_cross_entropy_balanced(labels=labels, logits=dev4)

                cv5_1 = slim.conv2d(cv5, num_outputs=1, kernel_size=1, stride=1,activation_fn=None)
                dev5 = slim.conv2d_transpose(cv5_1, num_outputs=1, kernel_size=32, stride=16,activation_fn=tf.nn.sigmoid)
                loss5 = sigmoid_cross_entropy_balanced(labels=labels, logits=dev5)

                ct = tf.concat([cv1_1,dev2,dev3,dev4,dev5],3)
                final_cv = slim.conv2d(ct,num_outputs=1,kernel_size=1,stride=1,weights_initializer=tf.constant_initializer(0.2),activation_fn=tf.nn.sigmoid)

                fuse_loss = sigmoid_cross_entropy_balanced(labels=labels, logits=final_cv)

                pred = tf.cast(tf.greater(final_cv, 0.5), tf.int32, name='predictions')
                ers = tf.cast(tf.not_equal(pred, tf.cast(labels, tf.int32)), tf.float32)
                ers = tf.reduce_mean(ers, name='pixel_error')

                return loss1*1+loss2*1+loss3*1+loss4*1+loss5*1+fuse_loss*1,ers,final_cv,svs

loss,error,out_put,vbs = create_model(image,label)
global_step  = tf.train.get_or_create_global_step()

lr = tf.train.exponential_decay(
    learning_rate=0.0002,
    global_step=global_step,
    decay_steps=500,
    decay_rate=0.5,
    staircase=True)

# Now we can define the optimizer that takes on the learning rate
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train_op = slim.learning.create_train_op(loss, optimizer)
saver = tf.train.Saver(vbs)

def restore_fn(sess):
    return saver.restore(sess, 'vgg_16.ckpt')


sv = tf.train.Supervisor(logdir='tree', summary_op=None, init_fn=None)

with sv.managed_session() as sess:
    ids = 0
    for step in range(1000000):
        im,em,ids = get_hed(ids)
        sess.run(train_op,feed_dict={image:im,label:em})
        if step % 10 ==0:
            ls,er,out,stp = sess.run([train_op,error,out_put,global_step],feed_dict={image:im,label:em})
            print stp,ls,er
        if step %500 ==0:
            op = np.squeeze(out, 0)
            op = np.squeeze(op,2)
            fig, ax = plt.subplots(1, 3, sharey=True, figsize=(12, 4))
            ax[0].imshow(im[0,:,:,:], aspect="auto")
            ax[1].imshow(em[0,:,:,0], aspect="auto", cmap='gray')
            ax[2].imshow(op, aspect="auto", cmap='gray')
            plt.show()


