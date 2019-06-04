import tensorflow as tf
from CasualConv1D import CausalConv1D

class ResidualBlock(tf.layers.Layer):
    def __init__(self, 
                filters,
                kernel_size,
                strides=1,
                dilation_rate=1,
                name=None,
                dtype=None,
                **kwargs):
        super(ResidualBlock, self).__init__(
            dtype=dtype,
            name=name, 
            **kwargs
        )
        self.initial_convolution = CausalConv1D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            dilation_rate=dilation_rate,
            name=name
        )
        self.tanh_Convolution = CausalConv1D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            dilation_rate=dilation_rate,
            name=name
        )
        self.sigmoid_Convolution = CausalConv1D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            dilation_rate=dilation_rate,
            name=name
        )
        self.conv1x1 = CausalConv1D(
            filters=filters,
            kernel_size=1,
            strides=strides,
            dilation_rate=1,
            name=name
        )
        self.skipConv = CausalConv1D(
            filters=filters,
            kernel_size=1,
            strides=strides,
            dilation_rate=1,
            name=name
        )
        self.skipConv1 = CausalConv1D(
            filters=filters,
            kernel_size=1,
            strides=strides,
            dilation_rate=1,
            name=name
        )
        self.skipConv2 = CausalConv1D(
            filters=filters,
            kernel_size=1,
            strides=strides,
            dilation_rate=1,
            name=name
        )
        

    def call(self, inputs, final_layer=False):
        x = self.tanh_Convolution(inputs)
        y = self.sigmoid_Convolution(inputs)
        tanh = tf.math.tanh(x)
        sigmoid = tf.math.sigmoid(y)
        multiply = tf.math.multiply(tanh, sigmoid)
        if final_layer:
            skip = self.skipConv(multiply)
            skip = tf.reduce_sum(skip, 2, keepdims=True)
            skip = tf.nn.relu(skip)
            skip = self.skipConv1(skip)
            skip = tf.nn.relu(skip)
            skip = self.skipConv2(skip)
            skip = tf.nn.softmax(skip)
            return skip
        conv = self.conv1x1(multiply)
        return conv + inputs

# def setupResBlock(filters, kernel_size, dilation_rate, name):
#     convs = {}
#     convs["tanh_Convolution"] = CausalConv1D(
#         filters=filters,
#         kernel_size=kernel_size,
#         strides=1,
#         dilation_rate=dilation_rate,
#         name=name
#     )
#     convs["sigmoid_Convolution"] = CausalConv1D(
#         filters=filters,
#         kernel_size=kernel_size,
#         strides=1,
#         dilation_rate=dilation_rate,
#         name=name
#     )
#     convs["conv1x1"] = CausalConv1D(
#         filters=filters,
#         kernel_size=1,
#         strides=1,
#         dilation_rate=1,
#         name=name
#     )
#     convs["skipConv"] = CausalConv1D(
#         filters=filters,
#         kernel_size=1,
#         strides=1,
#         dilation_rate=1,
#         name=name
#     )
#     return convs

# def resBlock(inputs, kernel_size, dilation_rate, name, final_layer=False):
#     filters = inputs.shape[2]
#     convs = setupResBlock(filter, kernel_size, dilation_rate, name)
#     x = self.tanh_Convolution(inputs)
#     y = self.sigmoid_Convolution(inputs)
#     tanh = tf.math.tanh(x)
#     sigmoid = tf.math.sigmoid(y)
#     multiply = tf.math.multiply(tanh, sigmoid)
#     if final_layer:
#         skip = self.skipConv(multiply)
#         return skip
#     conv = self.conv1x1(multiply)
#     return conv + inputs