from tensorflow.contrib import slim
import tensorflow as tf
from handler_data import get_hed
import numpy as np
from matplotlib import pyplot as plt
from inception_resnet_v2 import block8,block17,block35

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


def inception_resnet_v2_arg_scope(weight_decay=0.00004,
                                  batch_norm_decay=0.9997,
                                  batch_norm_epsilon=0.001):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_regularizer=slim.l2_regularizer(weight_decay)):
        batch_norm_params = {
            'decay': batch_norm_decay,
            'epsilon': batch_norm_epsilon,
        }
        # Set activation_fn and parameters for batch_norm.
        with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params) as scope:
            return scope


def bliliner_additive_upsampleing(featear,out_channel,stride):

    in_channel = featear.get_shape().as_list()[3]
    assert in_channel % out_channel == 0
    channel_split = in_channel/out_channel
    new_shape = featear.get_shape().as_list()
    new_shape[1] *= stride
    new_shape[2] *= stride
    new_shape[3] *= out_channel
    up_sample_feature = tf.image.resize_images(featear,new_shape[1:3])
    out_list = []
    for i in range(out_channel):
        splited_upsample = up_sample_feature[:,:,:,i*channel_split:(i+1)*channel_split]
        out_list.append(tf.reduce_sum(splited_upsample,axis=-1))
    fea = tf.stack(out_list,axis=-1)
    fea = slim.conv2d(fea,out_channel,kernel_size=stride*2,activation_fn=tf.nn.tanh)
    return fea

ck_reader = tf.train.NewCheckpointReader('/home/dsl/all_check/inception_resnet_v2_2016_08_30.ckpt')
for sd in ck_reader.get_variable_to_shape_map():
    print sd


image = tf.placeholder(dtype=tf.float32,shape=(1,512,512,3))
label = tf.placeholder(dtype=tf.float32,shape=(1,512,512,1))

def create_model(inputs, labels):
    with slim.arg_scope(inception_resnet_v2_arg_scope()):
        padding = 'SAME'
        with tf.variable_scope('InceptionResnetV2', 'InceptionResnetV2', [inputs]):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
                cv1 = slim.conv2d(inputs, 32, 3, stride=2, padding=padding,scope='Conv2d_1a_3x3')
                cv1 = slim.conv2d(cv1, 32, 3, padding=padding,scope='Conv2d_2a_3x3')
                cv1 = slim.conv2d(cv1, 64, 3, scope='Conv2d_2b_3x3')

                pool1 = slim.max_pool2d(cv1, 3, stride=2, padding=padding,scope='MaxPool_3a_3x3')

                cv2 = slim.conv2d(pool1, 80, 1, padding=padding,scope='Conv2d_3b_1x1')
                cv2 = slim.conv2d(cv2, 192, 3, padding=padding,scope='Conv2d_4a_3x3')

                pool2 = slim.max_pool2d(cv2, 3, stride=2, padding=padding,
                                      scope='MaxPool_5a_3x3')

                with tf.variable_scope('Mixed_5b'):
                    with tf.variable_scope('Branch_0'):
                        tower_conv = slim.conv2d(pool2, 96, 1, scope='Conv2d_1x1')
                    with tf.variable_scope('Branch_1'):
                        tower_conv1_0 = slim.conv2d(pool2, 48, 1, scope='Conv2d_0a_1x1')
                        tower_conv1_1 = slim.conv2d(tower_conv1_0, 64, 5,
                                                    scope='Conv2d_0b_5x5')
                    with tf.variable_scope('Branch_2'):
                        tower_conv2_0 = slim.conv2d(pool2, 64, 1, scope='Conv2d_0a_1x1')
                        tower_conv2_1 = slim.conv2d(tower_conv2_0, 96, 3,
                                                    scope='Conv2d_0b_3x3')
                        tower_conv2_2 = slim.conv2d(tower_conv2_1, 96, 3,
                                                    scope='Conv2d_0c_3x3')
                    with tf.variable_scope('Branch_3'):
                        tower_pool = slim.avg_pool2d(pool2, 3, stride=1, padding='SAME',
                                                     scope='AvgPool_0a_3x3')
                        tower_pool_1 = slim.conv2d(tower_pool, 64, 1,
                                                   scope='Conv2d_0b_1x1')
                    cv3 = tf.concat([tower_conv, tower_conv1_1, tower_conv2_2, tower_pool_1], 3)
                cv3 = slim.repeat(cv3, 10, block35, scale=0.17)



                with tf.variable_scope('Mixed_6a'):
                    with tf.variable_scope('Branch_0'):
                        tower_conv = slim.conv2d(cv3, 384, 3, stride=1 ,
                                                 padding=padding,
                                                 scope='Conv2d_1a_3x3')
                    with tf.variable_scope('Branch_1'):
                        tower_conv1_0 = slim.conv2d(cv3, 256, 1, scope='Conv2d_0a_1x1')
                        tower_conv1_1 = slim.conv2d(tower_conv1_0, 256, 3,
                                                    scope='Conv2d_0b_3x3')
                        tower_conv1_2 = slim.conv2d(tower_conv1_1, 384, 3,
                                                    stride=1 ,
                                                    padding=padding,
                                                    scope='Conv2d_1a_3x3')
                    with tf.variable_scope('Branch_2'):
                        tower_pool = slim.max_pool2d(cv3, 3, stride=1 ,
                                                     padding=padding,
                                                     scope='MaxPool_1a_3x3')
                    cv4 = tf.concat([tower_conv, tower_conv1_2, tower_pool], 3)


                with slim.arg_scope([slim.conv2d], rate=2 ):
                    cv4 = slim.repeat(cv4, 20, block17, scale=0.10)

                with tf.variable_scope('Mixed_7a'):
                    with tf.variable_scope('Branch_0'):
                        tower_conv = slim.conv2d(cv4, 256, 1, scope='Conv2d_0a_1x1')
                        tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2,
                                                   padding=padding,
                                                   scope='Conv2d_1a_3x3')
                    with tf.variable_scope('Branch_1'):
                        tower_conv1 = slim.conv2d(cv4, 256, 1, scope='Conv2d_0a_1x1')
                        tower_conv1_1 = slim.conv2d(tower_conv1, 288, 3, stride=2,
                                                    padding=padding,
                                                    scope='Conv2d_1a_3x3')
                    with tf.variable_scope('Branch_2'):
                        tower_conv2 = slim.conv2d(cv4, 256, 1, scope='Conv2d_0a_1x1')
                        tower_conv2_1 = slim.conv2d(tower_conv2, 288, 3,
                                                    scope='Conv2d_0b_3x3')
                        tower_conv2_2 = slim.conv2d(tower_conv2_1, 320, 3, stride=2,
                                                    padding=padding,
                                                    scope='Conv2d_1a_3x3')
                    with tf.variable_scope('Branch_3'):
                        tower_pool = slim.max_pool2d(cv4, 3, stride=2,
                                                     padding=padding,
                                                     scope='MaxPool_1a_3x3')
                    cv5 = tf.concat([tower_conv_1, tower_conv1_1, tower_conv2_2, tower_pool], 3)

                cv5 = slim.repeat(cv5, 9, block8, scale=0.20)
                cv5 = block8(cv5, activation_fn=None)

                cv5 = slim.conv2d(cv5, 1536, 1, scope='Conv2d_7b_1x1')
                print cv1
                print cv2
                print cv3
                print cv4
                print cv5


                svs = slim.get_variables_to_restore()



                cv1_1 = slim.conv2d(cv1,num_outputs=1,kernel_size=1,stride=1,activation_fn=tf.nn.tanh)
                dev1 = bliliner_additive_upsampleing(cv1_1, 1, 2)
                loss1 = sigmoid_cross_entropy_balanced(labels=labels,logits=dev1)


                cv2_1 = slim.conv2d(cv2,num_outputs=1,kernel_size=1,stride=1,activation_fn=None)
                dev2 = bliliner_additive_upsampleing(cv2_1,1,4)
                loss2 = sigmoid_cross_entropy_balanced(labels=labels, logits=dev2)

                cv3_1 = slim.conv2d(cv3, num_outputs=1, kernel_size=1, stride=1,activation_fn=None)
                dev3 = bliliner_additive_upsampleing(cv3_1,1,8)
                loss3 = sigmoid_cross_entropy_balanced(labels=labels, logits=dev3)

                cv4_1 = slim.conv2d(cv4, num_outputs=1, kernel_size=1, stride=1,activation_fn=None)
                dev4 = bliliner_additive_upsampleing(cv4_1,1,8)
                loss4 = sigmoid_cross_entropy_balanced(labels=labels, logits=dev4)

                cv5_1 = slim.conv2d(cv5, num_outputs=1, kernel_size=1, stride=1,activation_fn=None)
                dev5 = bliliner_additive_upsampleing(cv5_1,1,16)
                loss5 = sigmoid_cross_entropy_balanced(labels=labels, logits=dev5)

                ct = tf.concat([dev1,dev2,dev3,dev4,dev5],3)
                final_cv = slim.conv2d(ct,num_outputs=1,kernel_size=1,stride=1,weights_initializer=tf.constant_initializer(0.2),activation_fn=tf.nn.tanh)

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
    decay_steps=1000,
    decay_rate=0.7,
    staircase=True)

# Now we can define the optimizer that takes on the learning rate
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train_op = slim.learning.create_train_op(loss, optimizer)
saver = tf.train.Saver(vbs)

def restore_fn(sess):
    return saver.restore(sess, '/home/dsl/all_check/inception_resnet_v2_2016_08_30.ckpt')


sv = tf.train.Supervisor(logdir='resv2_fin', summary_op=None, init_fn=restore_fn)

with sv.managed_session() as sess:
    ids = 0
    for step in range(1000000):
        org_im,im,em,ids,name = get_hed(ids)
        sess.run(train_op,feed_dict={image:im,label:em})

        if step % 10 ==0:
            ls,er,out,stp = sess.run([train_op,error,out_put,global_step],feed_dict={image:im,label:em})
            print stp,ls,er
        if step %1000 ==0:
            op = np.squeeze(out, 0)
            op = np.squeeze(op,2)
            fig, ax = plt.subplots(1, 3, sharey=True, figsize=(12, 4))
            ax[0].imshow(org_im, aspect="auto")
            ax[1].imshow(em[0,:,:,0], aspect="auto", cmap='gray')
            ax[2].imshow(op, aspect="auto", cmap='gray')
            plt.show()


