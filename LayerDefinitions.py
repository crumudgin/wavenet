import tensorflow as tf

def casual_conv(matrix, weight_kernal, dilation_rate, name="casual_conv"):
    with tf.name_scope(name):
        shifted = tf.concat([[0], matrix], 1)
        return shifted
