import tensorflow as tf
import numpy as np
import wave
from PreprocessData import open_file

from AudioReader import *

class WaveNet:
    # def __init__(self, data):
    #     self.skip_sum = tf.zeros_like(data)
    #     self.data = data

    def dilated_causle_convolution(self, x, dilation_value, filter_out, filter_width=2, name=None):
        channels = x.shape[3]
        padding = (filter_width - 1) * dilation_value
        x = tf.pad(x, tf.constant([(0, 0,), (0, 0,), (1, 0), (0, 0)]) * padding)
        filters = tf.get_variable("%s_weights" %name, [1, filter_width, channels, filter_out])
        dilations = [1, 1, dilation_value, 1]
        x = tf.nn.conv2d(x, filters, [1, 1, 1, 1], "VALID", dilations=dilations, name="%s_conv" %name)
        return x

    def skip_connection(self, x, name="skip"):
        skip = self.dilated_causle_convolution(x, 1, 255, 1, name="%s_1x1_conv1" %name)
        skip = tf.reduce_sum(skip, 2, keepdims=True)
        skip = tf.nn.relu(skip)
        skip = self.dilated_causle_convolution(x, 1, 255, 1, name="%s_1x1_conv2" %name)
        skip = tf.nn.relu(skip)
        skip = self.dilated_causle_convolution(x, 1, 255, 1, name="%s_1x1_conv3" %name)
        return tf.nn.softmax(skip)

    def residual_block(self, x, dilation_value, out_channels=32, name=None):
        input_tensor = x
        tanh = self.dilated_causle_convolution(x, dilation_value, out_channels, name="%s_tanh_conv" %name)
        sigmoid = self.dilated_causle_convolution(x, dilation_value, out_channels, name="%s_sigmoid_conv" %name)

        tanh = tf.math.tanh(tanh)
        sigmoid = tf.math.sigmoid(sigmoid)

        x = tf.math.multiply(tanh, sigmoid)
        skip_conv = self.dilated_causle_convolution(x, 1, out_channels, 1, name="%s_skip_conv" %name)
        self.skip_sum = tf.add(self.skip_sum, skip_conv)
        x = self.dilated_causle_convolution(x, 1, out_channels, 1, name="%s_1x1_conv" %name)
        return input_tensor + x


    def model(self, data):
        x = self.dilated_causle_convolution(data, 1, 32, name="input_conv")
        self.skip_sum = tf.zeros_like(x)
        x = self.residual_block(x, 1, name="res_1")
        x = self.residual_block(x, 2, name="res_2")
        x = self.residual_block(x, 4, name="res_3")
        x = self.residual_block(x, 8, name="res_4")
        x = self.skip_connection(self.skip_sum)
        return x

waveNet = WaveNet()

data = open_file("p225_001.wav")

x = waveNet.model(data)

sess = tf.Session()
init = tf.global_variables_initializer()

sess.run(init)



print(sess.run(x))