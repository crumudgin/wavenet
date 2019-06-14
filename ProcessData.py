import librosa
import tensorflow as tf
import numpy as np
import wave

def mu_law_encode(audio, quantization_channels=256):
    '''
    Quantizes waveform amplitudes.
    All credit for this goes to ibab (https://github.com/ibab/tensorflow-wavenet)
    '''
    with tf.name_scope('encode'):
        mu = tf.to_float(quantization_channels - 1)
        # Perform mu-law companding transformation (ITU-T, 1988).
        # Minimum operation is here to deal with rare large amplitudes caused
        # by resampling.
        safe_audio_abs = tf.minimum(tf.cast(tf.abs(audio), tf.float32), 1.0)
        magnitude = tf.log1p(mu * safe_audio_abs) / tf.log1p(mu)
        signal = tf.cast(tf.sign(audio), tf.float32) * magnitude
        # Quantize signal to the specified number of levels.
        return tf.to_int32((signal + 1) / 2 * mu + 0.5)

def mu_law_decode(output, quantization_channels=256):
    '''
    Recovers waveform from quantized values.
    All credit for this goes to ibab (https://github.com/ibab/tensorflow-wavenet)
    '''
    with tf.name_scope('decode'):
        mu = quantization_channels - 1
        # Map values back to [-1, 1].
        signal = 2 * (tf.to_float(output) / mu) - 1
        # Perform inverse of mu-law transformation.
        magnitude = (1 / mu) * ((1 + mu)**abs(signal) - 1)
        return tf.sign(signal) * magnitude

def open_file(file):
	with tf.Session() as session:
		audio, sr = librosa.load(file, sr=None, mono=True)
		audio = audio.reshape(-1, 1)
		data = mu_law_encode(audio, 256)
		data = tf.one_hot(data, 256)
		data = tf.convert_to_tensor([data], dtype="float32")
		tf.global_variables_initializer().run()
		data = session.run(data)
	return data, sr

def write_to_file(data, sr, path="output.wav"):
	with tf.Session() as session:
		data = mu_law_decode(data)
		tf.global_variables_initializer().run()
		data = session.run(data)
		librosa.output.write_wav(path, data, sr)
	return data