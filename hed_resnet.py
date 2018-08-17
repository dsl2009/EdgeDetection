from tensorflow.contrib import slim
import tensorflow as tf
from handler_data_mult import q
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



def export(log_dir,output_dir):

    means = [123.68,116.78,103.94]
    input = tf.placeholder(tf.string, shape=[1])
    input_data = tf.decode_base64(input[0])
    input_image = tf.image.decode_jpeg(input_data)
    input_image = tf.image.resize_images(input_image,[256,256])
    input_image.set_shape([256,256,3])
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



def identity_block(input_tensor, in_depth,rate,scope):
    with tf.variable_scope(scope):
        x = slim.conv2d(input_tensor, num_outputs=in_depth, kernel_size=1, stride=1)
        x = slim.conv2d(x, num_outputs=in_depth, kernel_size=3, stride=1, rate=rate)
        x = slim.conv2d(x, num_outputs=in_depth * 4, kernel_size=1, stride=1,activation_fn=None)
        x += input_tensor
        return tf.nn.relu(x)



def conv_block(input_tensor, in_depth,  rate,stride=2):
    x = slim.conv2d(input_tensor, num_outputs=in_depth, kernel_size=1, stride=stride)
    x = slim.conv2d(x, num_outputs=in_depth, kernel_size=3, stride=1, rate=rate)
    x = slim.conv2d(x, num_outputs=in_depth * 4, kernel_size=1, stride=1, activation_fn=None)
    shortcut =  x = slim.conv2d(input_tensor, num_outputs=in_depth* 4, kernel_size=1, stride=stride,activation_fn=None)
    x +=shortcut
    return tf.nn.relu(x)



def bliliner_additive_upsampleing(featear,out_channel,stride):

    in_channel = featear.get_shape().as_list()[3]
    assert in_channel % out_channel == 0
    channel_split = in_channel/out_channel
    channel_split = tf.cast(channel_split, tf.int32)
    new_shape = featear.get_shape().as_list()
    new_shape[1] *= stride
    new_shape[2] *= stride
    new_shape[3] *= out_channel
    up_sample_feature = tf.image.resize_bilinear(featear,new_shape[1:3])
    out_list = []
    for i in range(out_channel):
        splited_upsample = up_sample_feature[:,:,:,i*channel_split:(i+1)*channel_split]
        out_list.append(tf.reduce_sum(splited_upsample,axis=-1))
    fea = tf.stack(out_list,axis=-1)
    #fea = slim.conv2d(fea,out_channel,kernel_size=stride*2,activation_fn=tf.nn.tanh)
    fea = slim.separable_conv2d(fea,out_channel,kernel_size=stride*2,depth_multiplier=4,activation_fn=tf.nn.tanh)
    return fea


image = tf.placeholder(dtype=tf.float32,shape=(8,256,256,3))
label = tf.placeholder(dtype=tf.float32,shape=(8,256,256,1))

def create_model(inputs, labels=None,is_train=True):
    with slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=0.99)):
        conv1 = slim.conv2d(inputs, num_outputs=64, kernel_size=7, stride=1)
        pol1 = slim.max_pool2d(conv1, kernel_size=3, stride=2)

        # Stage 2
        conv2 = conv_block(pol1, 64, rate=2, stride=1)
        conv2 = slim.repeat(conv2, 2, identity_block, scope='block1',in_depth=64, rate=2)
        # Stage 3
        conv3 = conv_block(conv2, 128, rate=2, stride=2)
        conv3 = slim.repeat(conv3, 3, identity_block, scope='block2',in_depth=128, rate=2)
        # Stage 4
        conv4 = conv_block(conv3, 256, rate=2, stride=2)
        conv4 = slim.repeat(conv4, 22, identity_block, scope='block3',in_depth=256, rate=2)
        # Stage 5
        conv5 = conv_block(conv4, 512, rate=2, stride=2)
        conv5 = slim.repeat(conv5, 2, identity_block,scope='block4', in_depth=512, rate=4)

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
            return loss1*1+loss2*1+loss3*1+loss4*1+loss5*1+fuse_loss*3,ers,final_cv,svs,dev2,dev5,cv1_1
        else:
            return final_cv

#export('tea','export_tea')
export(log_dir='land',output_dir='org')
loss,error,out_put,vbs,dv2,dv5,con3 = create_model(image,label)
tf.losses.add_loss(loss)
global_step  = tf.train.get_or_create_global_step()

lr = tf.train.exponential_decay(
    learning_rate=0.002,
    global_step=global_step,
    decay_steps=1000,
    decay_rate=0.9,
    staircase=True)

# Now we can define the optimizer that takes on the learning rate
total_loss = tf.losses.get_total_loss()
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train_op = slim.learning.create_train_op(total_loss, optimizer)

saver = tf.train.Saver(vbs)

def restore_fn(sess):
    return saver.restore(sess, 'vgg_16.ckpt')


sv = tf.train.Supervisor(logdir='land', summary_op=None, init_fn=None,save_model_secs=300)

with sv.managed_session() as sess:
    ids = 0
    for step in range(1000000):

            org_im,im,em = q.get()
            ls = sess.run(train_op,feed_dict={image:im,label:em})
            print(ls)
            if step % 1 ==0:
                ls,er,out,stp,v2,v5,v1 = sess.run([train_op,error,out_put,global_step,dv2,dv5,con3],feed_dict={image:im,label:em})

            if step %1==0:
                for s in range(8):
                    plt.subplot(221)
                    plt.title('original')
                    plt.imshow(org_im[s,:,:,:], aspect="auto")

                    plt.subplot(222)
                    plt.title('step1')
                    plt.imshow(v1[s,:,:,0], aspect="auto", cmap='gray')

                    plt.subplot(223)
                    plt.title('step2')
                    plt.imshow(v2[s,:,:,0], aspect="auto", cmap='gray')

                    plt.subplot(224)
                    plt.title('final')
                    plt.imshow(out[s,:,:,0], aspect="auto", cmap='gray')
                    plt.show()


