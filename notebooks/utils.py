## The following utility functions are used to convert a standard tensorflow type to a tf.train.Example-compatible tf.train.Feature

import tensorflow as tf

def _bytes_feature(value):
    'Returns a bytes-list from a string or a byte.'
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() #Byteslist wil not unpack a string from an eager tensor.
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def _float_feature(value):
    'Returns a float list from a float or a double.'
    return tf.train.Feature(float_list = tf.train.FloatList(value = [value]))

def _int64_feature(value):
    'Returns an int64 list from a bool/enum/int/uint.'
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))