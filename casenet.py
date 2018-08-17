import tensorflow as tf
from tensorflow.contrib import slim
from nets import resnet_v2
from handler_data import get_data
image_size = 352
batch_size = 1
num_class = 20

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



ig = tf.placeholder(shape=(1,image_size,image_size,3),dtype=tf.float32)
label = tf.placeholder(shape=(1,image_size,image_size,num_class),dtype=tf.float32)
beta = tf.placeholder(dtype=tf.float32)
def create_model():
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        conv1 = slim.conv2d(ig,num_outputs=64,kernel_size=7,stride=1)
        pol1 = slim.max_pool2d(conv1,kernel_size=3,stride=2)

        conv2 = slim.conv2d(pol1,num_outputs=256,kernel_size=1,activation_fn=None,rate=2)
        conv2 = resnet_v2_block(conv2, scope='block1', in_depth=64, add=conv2, is_downsample=True,rate=2)
        conv2 = slim.repeat(conv2,2,resnet_v2_block,scope='block1',in_depth=64)
        conv2 = tf.nn.relu(conv2)

        conv3 = slim.conv2d(conv2,num_outputs=512,kernel_size=1,stride=2,activation_fn=None,rate=2)
        conv3 = resnet_v2_block(conv3,scope='block2',in_depth=128,add=conv2,is_downsample=True,rate=2)
        conv3 = slim.repeat(conv3,3,resnet_v2_block,scope='block2',in_depth=128)
        conv3 = tf.nn.relu(conv3)

        conv4 = slim.conv2d(conv3,num_outputs=1024,kernel_size=1,stride=2,activation_fn=None,rate=2)
        conv4 = resnet_v2_block(conv4, scope='block3', in_depth=256, add=conv3, is_downsample=True,rate=2)
        conv4 = slim.repeat(conv4, 22, resnet_v2_block, scope='block3', in_depth=256)
        conv4 = tf.nn.relu(conv4)

        conv5 = slim.conv2d(conv4, num_outputs=2048, kernel_size=1, stride=1, activation_fn=None,rate=4)
        conv5 = resnet_v2_block(conv5, scope='block4', in_depth=512, add=conv4, is_downsample=False,rate=4)
        conv5 = slim.repeat(conv5, 2, resnet_v2_block, scope='block4', in_depth=512,rate=4)
        conv5 = tf.nn.relu(conv5)

        print conv5

        feature_side1 = slim.conv2d(conv1,num_outputs=1,kernel_size=1)

        feature_side2 = slim.conv2d(conv2, num_outputs=1, kernel_size=1)
        feature_side2 = slim.conv2d_transpose(feature_side2, num_outputs=1, kernel_size=4,stride=2)

        feature_side3 = slim.conv2d(conv3, num_outputs=1, kernel_size=1)
        feature_side3 = slim.conv2d_transpose(feature_side3, num_outputs=1, kernel_size=8, stride=4)

        cls_loss = slim.conv2d(conv5,num_outputs=num_class,kernel_size=1)
        cls_loss = slim.conv2d_transpose(cls_loss, num_outputs=num_class, kernel_size=16, stride=8)



        contact_feature = None
        for numclass in range(num_class):
            if contact_feature is None:
                contact_feature = tf.concat([tf.slice(cls_loss,[0,0,0,numclass],[-1, -1, -1, 1])
                                                ,feature_side1, feature_side2, feature_side3],axis=3)

            else:
                contact_feature = tf.concat([contact_feature, tf.slice(cls_loss, [0, 0, 0, numclass], [-1, -1, -1, 1])
                                                , feature_side1, feature_side2, feature_side3], axis=3)



        fea_loss = slim.conv2d(contact_feature,num_outputs=num_class,kernel_size=1)
        return cls_loss, fea_loss

c_loss, f_loss = create_model()

flat_c_loss = tf.reshape(c_loss,(-1,num_class))
flat_fea_loss = tf.reshape(f_loss,(-1,num_class))
flat_label = tf.reshape(label,(-1,num_class))

#closs = tf.reduce_mean(-beta*flat_label*tf.log(tf.sigmoid(flat_c_loss))-(1-beta)*(1-flat_label)*tf.log(1-tf.sigmoid(flat_c_loss)))
#floss = tf.reduce_mean(-beta*flat_label*tf.log(tf.sigmoid(flat_fea_loss))-(1-beta)*(1-flat_label)*tf.log(1-tf.sigmoid(flat_fea_loss)))
closs = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=flat_label,logits=flat_c_loss))
floss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=flat_label,logits=flat_fea_loss))
total_loss = closs+floss

opt = tf.train.AdamOptimizer(learning_rate=0.01).minimize(total_loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    idx = 0
    for s in range(100000):
        img, lab, bt ,idx = get_data(idx)
        bt = 0.5
        if img is None:
            continue


        sess.run(opt,feed_dict={ig:img,label:lab,beta:bt})
        print sess.run(total_loss,feed_dict={ig:img,label:lab,beta:bt})
