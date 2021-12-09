"""
Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Copyright 2020-2021 AWS DeepRacer Community. All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import cv2
import tensorflow.compat.v1 as tf
import numpy as np
from tensorflow.io.gfile import GFile


def load_session(pb_path, sensor='FRONT_FACING_CAMERA'):
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                            log_device_placement=True))
    print("load graph:", pb_path)
    with GFile(pb_path, 'rb') as f:
        graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    graph_nodes = [n for n in graph_def.node]
    names = []
    for t in graph_nodes:
        names.append(t.name)

    # For front cameras/stereo camera use the below
    x = sess.graph.get_tensor_by_name(
        'main_level/agent/main/online/network_0/{}/{}:0'.format(sensor, sensor))
    y = sess.graph.get_tensor_by_name('main_level/agent/main/online/network_1/ppo_head_0/policy:0')

    return sess, x, y


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def visualize_gradcam_discrete_ppo(
    sess,
    rgb_img,
    sensor='FRONT_FACING_CAMERA',
    category_index=0,
    num_of_actions=5
):
    '''
    @inp: model session, RGB Image - np array, action_index, total number of actions
    @return: overlayed heatmap
    '''

    img_arr = np.array(rgb_img)
    img_arr = rgb2gray(img_arr)
    img_arr = np.expand_dims(img_arr, axis=2)

    x = sess.graph.get_tensor_by_name(
        'main_level/agent/main/online/network_0/{}/{}:0'.format(sensor, sensor))
    y = sess.graph.get_tensor_by_name('main_level/agent/main/online/network_1/ppo_head_0/policy:0')
    feed_dict = {x: [img_arr]}

    # Get he policy head for clipped ppo in coach
    model_out_layer = sess.graph.get_tensor_by_name(
        'main_level/agent/main/online/network_1/ppo_head_0/policy:0')
    loss = tf.multiply(model_out_layer, tf.one_hot([category_index], num_of_actions))
    reduced_loss = tf.reduce_sum(loss[0])

    # For front cameras use the below
    conv_output = sess.graph.get_tensor_by_name(
        'main_level/agent/main/online/network_1/{}/Conv2d_4/Conv2D:0'.format(sensor))

    grads = tf.gradients(reduced_loss, conv_output)[0]
    output, grads_val = sess.run([conv_output, grads], feed_dict=feed_dict)
    weights = np.mean(grads_val, axis=(1, 2))
    cams = np.sum(weights * output, axis=3)

    # im_h, im_w = 120, 160##
    im_h, im_w = rgb_img.shape[:2]

    cam = cams[0]  # img 0
    image = np.uint8(rgb_img[:, :, ::-1] * 255.0)  # RGB -> BGR
    cam = cv2.resize(cam, (im_w, im_h))  # zoom heatmap
    cam = np.maximum(cam, 0)  # relu clip
    heatmap = cam / np.max(cam)  # normalize
    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)  # grayscale to color
    cam = np.float32(cam) + np.float32(image)  # overlay heatmap
    cam = 255 * cam / (np.max(cam) + 1E-5)  # Add expsilon for stability
    cam = np.uint8(cam)[:, :, ::-1]  # to RGB

    return cam
