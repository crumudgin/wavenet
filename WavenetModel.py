import tensorflow as tf
import numpy as np
import wave
from CasualConv1D import CausalConv1D
from ResidualBlock import ResidualBlock

from AudioReader import *

def model(data):
    res = ResidualBlock(255, 2)
    res2 = ResidualBlock(255, 2, dilation_rate=2)
    res3 = ResidualBlock(255, 2, dilation_rate=4)
    res4 = ResidualBlock(255, 2, dilation_rate=8)
    conv = CausalConv1D(255, 2) #TODO figure out how to do like 10 layers of this
    x = conv(data)
    x = res(x)
    x = res2(x)
    x = res3(x)
    x = res4(x, True) #TODO use skip connection instead of just conv
    return x

with wave.open("p225_001.wav") as audio:
    num_of_frames = audio.getnframes()
    frames = [i for i in audio.readframes(num_of_frames)]
    data = tf.one_hot(frames, 255)

data = tf.convert_to_tensor([data])

x = model(data)

sess = tf.Session()
init = tf.global_variables_initializer()

sess.run(init)



print(sess.run(x).shape)
# print(data.shape)