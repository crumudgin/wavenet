import tensorflow as tf
import numpy as np
import wave
from CasualConv1D import CausalConv1D
from ResidualBlock import ResidualBlock

from AudioReader import *

def dilated_causle_convolution(x, dilation_value, filter_out, filter_width=2, name=None):
    channels = x.shape[3]
    padding = (filter_width - 1) * dilation_value
    x = tf.pad(x, tf.constant([(0, 0,), (0, 0,), (1, 0), (0, 0)]) * padding)
    filters = tf.get_variable("%s_weights" %name, [1, filter_width, channels, filter_out])
    dilations = [1, 1, dilation_value, 1]
    x = tf.nn.conv2d(x, filters, [1, 1, 1, 1], "VALID", dilations=dilations, name="%s_conv" %name)
    return x

def skip_connection(x, out_channels, name):
    skip = dilated_causle_convolution(x, 1, max(out_channels * 20, 255), 1, name="%s_1x1_conv1" %name)
    skip = tf.reduce_sum(skip, 2, keepdims=True)
    skip = tf.nn.relu(skip)
    skip = dilated_causle_convolution(x, 1, 255, 1, name="%s_1x1_conv2" %name)
    skip = tf.nn.relu(skip)
    skip = dilated_causle_convolution(x, 1, 255, 1, name="%s_1x1_conv3" %name)
    return tf.nn.softmax(skip)

def residual_block(x, dilation_value, out_channels=0, final_block=False, name=None):
    in_channels = x.shape[3]
    out_channels = in_channels * 5 if out_channels == 0 else out_channels

    x = dilated_causle_convolution(x, 1, out_channels, name="%s_input_conv" %name)

    tanh = dilated_causle_convolution(x, dilation_value, out_channels, name="%s_tanh_conv" %name)
    sigmoid = dilated_causle_convolution(x, dilation_value, out_channels, name="%s_sigmoid_conv" %name)

    tanh = tf.math.tanh(tanh)
    sigmoid = tf.math.sigmoid(sigmoid)

    multiplied = tf.math.multiply(tanh, sigmoid)
    if final_block:
        return skip_connection(x, out_channels, name)
    multiplied = dilated_causle_convolution(x, 1, out_channels, 1, name="%s_1x1_conv" %name)
    return multiplied + x


def model(data):
    x = residual_block(data, 1, out_channels=32, name="res_1")
    x = residual_block(data, 2, name="res_2")
    x = residual_block(data, 4, name="res_3")
    x = residual_block(data, 8, final_block=True, name="res_4")
    return x

with wave.open("p225_001.wav") as audio:
    num_of_frames = audio.getnframes()
    frames = [i for i in audio.readframes(num_of_frames)]
    data = tf.one_hot(frames, 256) #I think I'm supposed to do mu encoding here

data = tf.convert_to_tensor([[data]], dtype="float32")
print(data.shape)

x = model(data)

sess = tf.Session()
init = tf.global_variables_initializer()

sess.run(init)



print(sess.run(x))