import tensorflow as tf
import json
import base64
import numpy as np
import cv2
import time
image_url = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/edge_detect/tea/org/42.jpg'
sess_config = tf.ConfigProto(allow_soft_placement=True)
sess_config.gpu_options.per_process_gpu_memory_fraction = 0.5


img = cv2.imread('ss.jpg',0)

edges = cv2.Canny(img, 100, 200)

image, contours, her = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cont in contours[0]:



def new_gen_edge(org,ip,name):

    org = cv2.imread(org)
    img = cv2.imread(ip,0)
    img = cv2.resize(img, (512, 512),interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(img, 100, 200)

    image, contours, her = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    cv2.drawContours(org, contours, -1, (0, 0, 255), 1)

    cv2.imwrite(name,org)



with tf.Session() as sess:
    saver = tf.train.import_meta_graph("export_tea/export.meta")
    saver.restore(sess, "export_tea/export")
    input_vars = json.loads(tf.get_collection("inputs")[0])
    output_vars = json.loads(tf.get_collection("outputs")[0])
    input = tf.get_default_graph().get_tensor_by_name(input_vars["input"])
    output = tf.get_default_graph().get_tensor_by_name(output_vars["output"])
    with open(image_url, "rb") as f:
        input_data = f.read()

    input_instance = dict(input=base64.urlsafe_b64encode(input_data).decode("ascii"), key="0")
    input_instance = json.loads(json.dumps(input_instance))

    input_value = np.array(input_instance["input"])
    output_value = sess.run(output, feed_dict={input: np.expand_dims(input_value, axis=0)})[0]

    output_instance = dict(output=output_value.decode("ascii"), key="0")

    b64data = output_instance["output"]
    b64data += "=" * (-len(b64data) % 4)
    output_data = base64.urlsafe_b64decode(b64data.encode("ascii"))
    with open('ss.jpg', "wb") as f:
        f.write(output_data)
    new_gen_edge(image_url,'ss.jpg','gen.jpg')
