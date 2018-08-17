from tensorflow.contrib import slim
import tensorflow as tf
from handler_data_mult_test import get_hed
import numpy as np
from matplotlib import pyplot as plt
from nets import resnet_v2
import json
import os
import cv2
import json
import base64
import glob
def draw():
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.3
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("org/export.meta")
        saver.restore(sess, "org/export")
        input_vars = json.loads(tf.get_collection("inputs")[0])
        output_vars = json.loads(tf.get_collection("outputs")[0])
        input = tf.get_default_graph().get_tensor_by_name(input_vars["input"])
        output = tf.get_default_graph().get_tensor_by_name(output_vars["output"])
        i = 0
        for ix,s in enumerate(glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/edge_detect/building/org/*.jpg')):
            print(s)
            with open(s, "rb") as f:
                input_data = f.read()

                input_instance = dict(input=base64.urlsafe_b64encode(input_data).decode("ascii"), key="0")
                input_instance = json.loads(json.dumps(input_instance))

                input_value = np.array(input_instance["input"])
                output_value = sess.run(output, feed_dict={input: np.expand_dims(input_value, axis=0)})[0]

                output_instance = dict(output=output_value.decode("ascii"), key="0")

                b64data = output_instance["output"]
                b64data += "=" * (-len(b64data) % 4)
                output_data = base64.urlsafe_b64decode(b64data.encode("ascii"))

                with open('land/'+str(ix)+'.jpg', "wb") as f1:
                    f1.write(output_data)
                    f1.flush()



draw()