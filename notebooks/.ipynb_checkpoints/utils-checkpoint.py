## The following utility functions are used to convert a standard tensorflow type to a tf.train.Example-compatible tf.train.Feature

import os
import tensorflow as tf

height = 720
width = 1280
num_depth = 3
video_frame_path = os.path.join('../inputs', 'video')


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

def _parse_image_function(example_proto):
    return tf.io.parse_single_example(example_proto, image_feature_description)

def _parse_video_function(frame_path):
    image_seq = []
    global num_depth
    
    frames_num = len(os.listdir(frame_path))
    
    for image_count in range(frames_num - 1):
        image_path = os.path.join(frame_path, f'{image_count}.jpg')
        
        feature_dict = {image_path : tf.io.FixedLenFeature([], tf.string),
                        'height': tf.io.FixedLenFeature([], tf.int64 ,default_value=0),
                        'width': tf.io.FixedLenFeature([], tf.int64 ,default_value=0),
                        'depth': tf.io.FixedLenFeature([], tf.int64 ,default_value=0)
                        }
        features = tf.io.parse_single_example(image_path, 
                                              features = feature_dict)
        
        image_buffer = tf.reshape(features[image_path], shape = [])
        
        image = tf.image.decode_jpeg(image_buffer, channels = num_depth)
        image = tf.reshape(image, tf.stack([height, width, num_depth]))
        image = tf.reshape(image, [1, height, width, num_depth])
        image_seq.append(image)
        
    image_seq = tf.concat(image_buffer, 0)
    image_seq = bytes(image_seq)
   
    
    return tf.train.Example(features = tf.train.Features(feature = feature_dict))