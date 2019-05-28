import tensorflow as tf
from CasualConv1D import CausalConv1D

class ResidualBlock(tf.layers.Layer):
    def __init__(self, 
                filters,
                kernel_size,
                strides=1,
                dilation_rate=1,
                activation=None,
                use_bias=True,
                kernel_initializer=None,
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                trainable=True,
                name=None,
                dtype=None,
                **kwargs):
        super(ResidualBlock, self).__init__(
            trainable=trainable,
            dtype=dtype,
            activity_regularizer=activity_regularizer,
            name=name, 
            **kwargs
        )
        self.tanh_Convolution = CausalConv1D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            trainable=trainable,
            name=name,
        )
        self.sigmoid_Convolution = CausalConv1D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            trainable=trainable,
            name=name,
        )
        self.conv1x1 = CausalConv1D(
            filters=filters,
            kernel_size=1,
            strides=strides,
            dilation_rate=1,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            trainable=trainable,
            name=name,
        )
        
       
    def build(self, input_shape):
        pass
        

    def call(self, inputs, final_layer=False):
        x = self.tanh_Convolution(inputs)
        y = self.sigmoid_Convolution(inputs)
        tanh = tf.math.tanh(x)
        sigmoid = tf.math.sigmoid(y)
        multiply = tf.math.multiply(tanh, sigmoid)
        conv = self.conv1x1(multiply)
        if final_layer:
            return conv
        return conv + inputs