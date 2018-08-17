from tensorflow.contrib import slim
import tensorflow as tf
from handler_data import get_hed,get_test_data
import numpy as np
from matplotlib import pyplot as plt
from nets import resnet_v2
import json
import os


def sigmoid_cross_entropy_balanced(logits, labels, name='cross_entropy_loss'):
    """
    Implements Equation [2] in https://arxiv.org/pdf/1504.06375.pdf
    Compute edge pixels for each training sample and set as pos_weights to
    tf.nn.weighted_cross_entropy_with_logits
    """
    y = tf.cast(labels, tf.float32)

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



def export(log_dir,output_dir):

    means = [123.68,116.78,103.94]
    input = tf.placeholder(tf.string, shape=[1])
    input_data = tf.decode_base64(input[0])
    input_image = tf.image.decode_jpeg(input_data)
    input_image = tf.image.resize_images(input_image,[512,512])
    input_image.set_shape([512,512,3])

    num_channels = input_image.get_shape().as_list()[-1]
    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=input_image)
    for i in range(num_channels):
        channels[i] -= means[i]
    img = tf.concat(axis=2, values=channels)
    img = tf.to_float(img)
    batch_input = tf.expand_dims(input_image, axis=0)


    batch_output = create_model(batch_input,is_train=False)

    output_image = tf.image.convert_image_dtype(batch_output, dtype=tf.uint8)[0]

    output_data = tf.image.encode_jpeg(output_image, quality=80)

    output = tf.convert_to_tensor([tf.encode_base64(output_data)])

    key = tf.placeholder(tf.string, shape=[1])
    inputs = {
        "key": key.name,
        "input": input.name
    }
    tf.add_to_collection("inputs", json.dumps(inputs))
    outputs = {
        "key": tf.identity(key).name,
        "output": output.name,
    }
    tf.add_to_collection("outputs", json.dumps(outputs))

    init_op = tf.global_variables_initializer()
    restore_saver = tf.train.Saver()
    export_saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)
        print("loading model from checkpoint")
        checkpoint = tf.train.latest_checkpoint(log_dir)
        restore_saver.restore(sess, checkpoint)
        print("exporting model")
        export_saver.export_meta_graph(filename=os.path.join(output_dir, "export.meta"))
        export_saver.save(sess, os.path.join(output_dir, "export"), write_meta_graph=False)

    return


def resnet_v2_block(inputs,scope='block1', in_depth=64,add=None,is_downsample=False,rate=2):
    with tf.variable_scope(scope):
        orig_x = inputs
        x = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')

        if add is not None:
            x = add
        with tf.variable_scope('sub1'):
            if is_downsample:
                x = slim.conv2d(x,num_outputs=in_depth,kernel_size=1,stride=2)
            else:
                x = slim.conv2d(x, num_outputs=in_depth, kernel_size=1, stride=1)

        with tf.variable_scope('sub2'):
            x = slim.conv2d(x, num_outputs=in_depth, kernel_size=3, stride=1,rate=rate)

        with tf.variable_scope('sub3'):
            x = slim.conv2d(x, num_outputs=in_depth*4, kernel_size=1, stride=1)

        with tf.variable_scope('sub_add'):


             x += orig_x


    return x

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


image = tf.placeholder(dtype=tf.float32,shape=(1,512,512,3))


def create_model(inputs, labels=None,is_train=True):
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        with slim.arg_scope([slim.conv2d],trainable = is_train):
            conv1 = slim.conv2d(inputs,num_outputs=64,kernel_size=7,stride=1)
            pol1 = slim.max_pool2d(conv1,kernel_size=3,stride=2)

            conv2 = slim.conv2d(pol1,num_outputs=256,kernel_size=1,activation_fn=None,rate=2)
            conv2 = resnet_v2_block(conv2, scope='block1', in_depth=64, add=conv2, is_downsample=False,rate=2)
            conv2 = slim.repeat(conv2,2,resnet_v2_block,scope='block1',in_depth=64)
            conv2 = tf.nn.relu(conv2)

            conv3 = slim.conv2d(conv2,num_outputs=512,kernel_size=1,stride=2,activation_fn=None)
            conv3 = resnet_v2_block(conv3,scope='block2',in_depth=128,add=conv2,is_downsample=True,rate=2)
            conv3 = slim.repeat(conv3,3,resnet_v2_block,scope='block2',in_depth=128)
            conv3 = tf.nn.relu(conv3)

            conv4 = slim.conv2d(conv3,num_outputs=1024,kernel_size=1,stride=2,activation_fn=None)
            conv4 = resnet_v2_block(conv4, scope='block3', in_depth=256, add=conv3, is_downsample=True,rate=2)
            conv4 = slim.repeat(conv4, 22, resnet_v2_block, scope='block3', in_depth=256)
            conv4 = tf.nn.relu(conv4)

            conv5 = slim.conv2d(conv4, num_outputs=2048, kernel_size=1, stride=2, activation_fn=None)
            conv5 = resnet_v2_block(conv5, scope='block4', in_depth=512, add=conv4, is_downsample=True,rate=4)
            conv5 = slim.repeat(conv5, 2, resnet_v2_block, scope='block4', in_depth=512,rate=4)
            conv5 = tf.nn.relu(conv5)

            svs = slim.get_variables_to_restore()

            cv1_1 = slim.conv2d(conv1, num_outputs=1, kernel_size=1, stride=1, activation_fn=tf.nn.tanh)

            cv2_1 = slim.conv2d(conv2, num_outputs=1, kernel_size=1, stride=1, activation_fn=None)
            dev2 = bliliner_additive_upsampleing(cv2_1, 1, 2)

            cv3_1 = slim.conv2d(conv3, num_outputs=1, kernel_size=1, stride=1, activation_fn=None)
            dev3 = bliliner_additive_upsampleing(cv3_1, 1, 4)

            cv4_1 = slim.conv2d(conv4, num_outputs=1, kernel_size=1, stride=1, activation_fn=None)
            dev4 = bliliner_additive_upsampleing(cv4_1, 1, 8)

            cv5_1 = slim.conv2d(conv5, num_outputs=1, kernel_size=1, stride=1, activation_fn=None)
            dev5 = bliliner_additive_upsampleing(cv5_1, 1, 16)

            ct = tf.concat([cv1_1, dev2, dev3, dev4, dev5], 3)
            final_cv = slim.conv2d(ct,num_outputs=1,kernel_size=1,stride=1,weights_initializer=tf.constant_initializer(0.2),activation_fn=tf.nn.tanh)
            if is_train:
                loss1 = sigmoid_cross_entropy_balanced(labels=labels, logits=cv1_1)
                loss2 = sigmoid_cross_entropy_balanced(labels=labels, logits=dev2)
                loss3 = sigmoid_cross_entropy_balanced(labels=labels, logits=dev3)
                loss4 = sigmoid_cross_entropy_balanced(labels=labels, logits=dev4)
                loss5 = sigmoid_cross_entropy_balanced(labels=labels, logits=dev5)
                fuse_loss = sigmoid_cross_entropy_balanced(labels=labels, logits=final_cv)
                pred = tf.cast(tf.greater(final_cv, 0.5), tf.int32, name='predictions')
                ers = tf.cast(tf.not_equal(pred, tf.cast(labels, tf.int32)), tf.float32)
                ers = tf.reduce_mean(ers, name='pixel_error')
                return loss1*1+loss2*1+loss3*1+loss4*1+loss5*1+fuse_loss*3,ers,final_cv,svs
            else:
                return final_cv


out_put = create_model(image,is_train=False)


sv = tf.train.Supervisor(logdir='tea', summary_op=None, init_fn=None)

with sv.managed_session() as sess:
    ids = 0
    for step in range(1000000):
            im = get_test_data('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/edge_detect/tea/org/100.jpg')
            out = sess.run(out_put,feed_dict={image:im})
            op = np.squeeze(out, 0)
            op = np.squeeze(op,2)

            plt.imshow(op,  cmap='gray')
            plt.show()


