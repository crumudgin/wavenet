import tensorflow as tf
import numpy as np
import wave
from ProcessData import open_file, write_to_file

from AudioReader import *

class WaveNet:
    def __init__(self):
        self.graph = tf.Graph()
        self.data = tf.placeholder(tf.float32, shape=(1, 98473, 1, 256))
        self.skip_sum = None

    def dilated_causle_convolution(self, x, dilation_value, filter_out, filter_width=2, name=None):
        channels = x.shape[3]
        padding = (filter_width - 1) * dilation_value
        x = tf.pad(x, tf.constant([(0, 0,), (0, 0,), (1, 0), (0, 0)]) * padding)
        filters = tf.get_variable("%s_weights" %name, [1, filter_width, channels, filter_out])
        dilations = [1, 1, dilation_value, 1]
        x = tf.nn.conv2d(x, filters, [1, 1, 1, 1], "VALID", dilations=dilations, name="%s_conv" %name)
        return x

    def skip_connection(self, x, name="skip"):
        skip = self.dilated_causle_convolution(x, 1, 256, 1, name="%s_1x1_conv1" %name)
        skip = tf.reduce_sum(skip, 2, keepdims=True)
        skip = tf.nn.relu(skip)
        skip = self.dilated_causle_convolution(skip, 1, 256, 1, name="%s_1x1_conv2" %name)
        skip = tf.nn.relu(skip)
        skip = self.dilated_causle_convolution(skip, 1, 256, 1, name="%s_1x1_conv3" %name)
        return tf.nn.softmax(skip)

    def residual_block(self, x, dilation_value, out_channels=32, name=None):
        input_tensor = x
        tanh = self.dilated_causle_convolution(x, dilation_value, out_channels, name="%s_tanh_conv" %name)
        sigmoid = self.dilated_causle_convolution(x, dilation_value, out_channels, name="%s_sigmoid_conv" %name)

        tanh = tf.math.tanh(tanh)
        sigmoid = tf.math.sigmoid(sigmoid)

        x = tf.math.multiply(tanh, sigmoid)
        skip_conv = self.dilated_causle_convolution(x, 1, out_channels, 1, name="%s_skip_conv" %name)
        skip_conv = tf.slice(skip_conv, [0, 360, 0, 0], [-1, skip_conv.shape[1] - 360, -1, -1])
        if self.skip_sum is None:
            self.skip_sum = skip_conv
        else:
            self.skip_sum += skip_conv
        self.skip_sum = tf.add(self.skip_sum, skip_conv)
        x = self.dilated_causle_convolution(x, 1, out_channels, 1, name="%s_1x1_conv" %name)
        return input_tensor + x


    def model(self):
        x = self.dilated_causle_convolution(self.data, 1, 32, name="input_conv")
        x = self.residual_block(x, 1, name="res_1")
        x = self.residual_block(x, 2, name="res_2")
        x = self.residual_block(x, 4, name="res_3")
        x = self.residual_block(x, 8, name="res_4")
        x = self.residual_block(x, 16, name="res_5")
        x = self.skip_connection(self.skip_sum)
        shape = self.data.shape
        usable_data = tf.slice(self.data, [0, 360, 0, 0], [-1, shape[1] - 360, -1, -1])
        print(x.shape)
        print(usable_data.shape)
        print(self.data.shape)
        loss = tf.losses.softmax_cross_entropy(usable_data, x)
        return x, loss


loaded_wav, sr = open_file("p225_001.wav")

waveNet = WaveNet()

with tf.Session() as session:
    

    
    x, loss = waveNet.model()

    optimizer = tf.train.AdamOptimizer().minimize(loss)

    tf.global_variables_initializer().run()
    print("initialized")

    counter = 100
    while counter > 0:
        output = session.run([optimizer, loss, x], feed_dict={waveNet.data: loaded_wav})
        counter -= 1
        print(counter)

waveform = [max(i[0]) for i in output[2][0]]
print(waveform)
write_to_file(waveform, sr)