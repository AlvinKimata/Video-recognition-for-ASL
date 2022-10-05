from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unit_3d
import sonnet as snt
import tensorflow as tf


class InceptionI3d(snt.Module):
    '''Inception I3D architecture.'''

    #Second return value.
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv2d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions'
    )

    def __init__(self, num_classes = 400, spatial_squeeze = True, 
    final_endpoint = 'Logits', name = 'inception_i3d'):
        '''
        Initializes a I3D model instance.

        Args:
            num_classes: The number of outputs in the logit layer which matches the Kinetics dataset.

            spatial_squeeze: Whether to squeeze the spatial dimensions for the logits before returning default (True).

            final_endpoint: The model contains many possible endpoints. It specifies the last 
            endpoint for the model to be built up to. `final_endpoint` will also be returned, in 
            a dictionary.

            name: A string, the name of this module.
        
        Raises:
            ValueError: If `final_endpoint` is not recognized.
        '''

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint is: %s' % final_endpoint)

        super(InceptionI3d, self).__init__(name = name)
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint

    def _build(self, inputs, is_training, dropout_keep_prob = 1.0):
        '''
        Connects the model to the inputs.

        Args:
            inputs: Inputs to the model, which should have dimensions
                `batch_size` x `num_frames` x 224 x 224 x `num_channels`.
            
            is_training: Whether to use training mode for `snt.BatchNorm` (boolean)

            dropout_keep_prob: Probability for the `tf.nn.dropout` layer 

        Returns:
            A tuple containing:
                1. Network output at location `self._final_endpoint`.
                2. A dictionary containing all endpoints up to `self._final_endpoint`, indexed by the endpoint name.

        Returns:
            ValueError: if `self._final_endpoint` is not recognized.
        '''

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' %self._final_endpoint)


        net = inputs
        end_points = {}
        end_point = 'Conv3d_1a_7x7'
        net = unit_3d.Unit3D(output_channels=64, kernel_shape=[7, 7, 7],
            stride = [2, 2, 2], name = end_point)(net, is_training = is_training)

        end_points[end_point] = net

        if self._final_endpoint == end_point: 
            return net, end_points
        

        #Max pooling layer.
        end_point = 'MaxPool3d_2a_3x3'
        net = tf.nn.max_pool3d(net, ksize=[1, 1, 3, 3, 1],
            strides = [1, 1, 2, 2, 1], padding = snt.pad.same, name = end_point)
        end_points[end_point] = net

        if self._final_endpoint == end_point: 
            return net, end_points

        #Second Conv3D block.
        end_point = 'Conv3d_2b_1x1'
        net = unit_3d.Unit3D(output_channels=64, kernel_shape = [1, 1, 1],
            name = end_point)(net, is_training = is_training)

        end_points[end_point] = net

        if self._final_endpoint == end_point:
            return net, end_points

        #Third Conv3D block.
        end_point = 'Conv2d_2c_3x3'
        net = unit_3d.Unit3D(output_channels=192, kernel_shape=[3, 3, 3],
        name = end_point)(net, is_training = is_training)
        end_points[end_point] = net

        if self._final_endpoint == end_point:
            return net, end_points

        #Second MaxPooling layer.
        end_point = 'MaxPool3d_3a_3x3'
        net = tf.nn.max_pool3d(net, ksize = [1, 1, 3, 3, 1], 
            strides = [1, 1, 2, 2, 1], padding = snt.pad.same, name = end_point)

        end_points[end_point] = net

        if self._final_endpoint == end_point:
            return net, end_points

        #First Mixed layer.
        end_point = 'Mixed_3b'

        with tf.compat.v1.variable_scope(end_point):
            with tf.compat.v1.variable_scope('Branch_0'):
                branch_0 = unit_3d.Unit3D(output_channels=64, kernel_shape = [1, 1, 1],
                    name = 'Conv3d_0a_1x1')(net, is_training = is_training)

            with tf.compat.v1.variable_scope('Branch_1'):
                branch_1 = unit_3d.Unit3D(output_channels=96, kernel_shape = [1, 1, 1],
                    name = 'Conv3d_0a_3x3')(net, is_training = is_training)
                
                branch_1 = unit_3d.Unit3D(output_channels=128, kernel_shape = [3, 3, 3],
                    name = 'Conv3d_0b_3x3')(branch_1, is_training = is_training)

            with tf.compat.v1.variable_scope('Branch_2'):
                branch_2 = unit_3d.Unit3D(output_channels=16, kernel_shape=[1, 1, 1],
                    name = 'Conv3d_0a_1x1')(net, is_training = is_training)

                branch_2 = unit_3d.Unit3D(output_channels=32, kernel_shape = [3, 3, 3],
                    name = 'Conv3d_0b_3x3')(branch_2, is_training = is_training)
            
            with tf.compat.v1.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize = [1, 3, 3, 3, 1],
                    strides = [1, 1, 1, 1, 1], padding = snt.pad.same, name = 'MaxPool3d_0a_3x3')

                branch_3 = unit_3d.Unit3D(output_channels=32, kernel_shape = [1, 1, 1],
                    name = 'Conv3d_0b_1x1')(branch_3, is_training = is_training)

            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)

        end_points[end_point] = net

        if self._final_endpoint == end_point:
            return net, end_points

        #Second Mixed layer.
        end_point = 'Mixed_3c'

        with tf.compat.v1.variable_scope(end_point):
            with tf.compat.v1.variable_scope('Branch_0'):
                branch_0 = unit_3d.Unit3D(output_channels=128, kernel_shape = [1, 1, 1],
                    name = 'Conv3d_0a_1x1')(net, is_training = is_training)

            with tf.compat.v1.variable_scope('Branch_1'):
                branch_1 = unit_3d.Unit3D(output_channels=128, kernel_shape = [1, 1, 1],
                    name = 'Conv3d_0z_1x1')(net, is_training = is_training)
                
                branch_1 = unit_3d.Unit3D(output_channels=192, kernel_shape = [3, 3, 3],
                    name = 'Conv3d_0b_3x3')(branch_1, is_training = is_training)

            with tf.compat.v1.variable_scope('Branch_2'):
                branch_2 = unit_3d.Unit3D(output_channels=32, kernel_shape = [1, 1, 1],
                    name = 'Conv3d_0a_1x1')(net, is_training = is_training)

                branch_2 = unit_3d.Unit3D(output_channels=96, kernel_shape = [3, 3, 3],
                    name = 'Conv3d_0b_3x3')(branch_2, is_training = is_training)
            
            with tf.compat.v1.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize = [1, 3, 3, 3, 1],
                    strides = [1, 1, 1, 1, 1], padding = snt.pad.same, name = 'MaxPool3d_0a_3x3')

                branch_3 = unit_3d.Unit3D(output_channels=32, kernel_shape = [1, 1, 1],
                    name = 'Conv3d_0b_1x1')(branch_3, is_training = is_training)

            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)

        end_points[end_point] = net

        if self._final_endpoint == end_point:
            return net, end_points

        end_point = 'Mixed_4b'
        with tf.compat.v1.variable_scope(end_point):
            with tf.compat.v1.variable_scope('Branch_0'):
                branch_0 = unit_3d.Unit3D(output_channels=192, kernel_shape = [1, 1, 1],
                    name = 'Conv3d_0a_1x1')(net, is_training = is_training)

            with tf.compat.v1.variable_scope('Branch_1'):
                branch_1 = unit_3d.Unit3D(output_channels=96, kernel_shape = [1, 1, 1],
                    name = 'Conv3d_0z_1x1')(net, is_training = is_training)
                
                branch_1 = unit_3d.Unit3D(output_channels=208, kernel_shape = [3, 3, 3],
                    name = 'Conv3d_0b_3x3')(branch_1, is_training = is_training)

            with tf.compat.v1.variable_scope('Branch_2'):
                branch_2 = unit_3d.Unit3D(output_channels=16, kernel_shape = [1, 1, 1],
                    name = 'Conv3d_0a_1x1')(net, is_training = is_training)

                branch_2 = unit_3d.Unit3D(output_channels=48, kernel_shape = [3, 3, 3],
                    name = 'Conv3d_0b_3x3')(branch_2, is_training = is_training)
            
            with tf.compat.v1.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize = [1, 3, 3, 3, 1],
                    strides = [1, 1, 1, 1, 1], padding = snt.pad.same, name = 'MaxPool3d_0a_3x3')

                branch_3 = unit_3d.Unit3D(output_channels=64, kernel_shape = [1, 1, 1],
                    name = 'Conv3d_0b_1x1')(branch_3, is_training = is_training)

            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)

        end_points[end_point] = net

        if self._final_endpoint == end_point:
            return net, end_points

        end_point = 'Mixed_4c'
        with tf.compat.v1.variable_scope(end_point):
            with tf.compat.v1.variable_scope('Branch_0'):
                branch_0 = unit_3d.Unit3D(output_channels=160, kernel_shape = [1, 1, 1],
                    name = 'Conv3d_0a_1x1')(net, is_training = is_training)

            with tf.compat.v1.variable_scope('Branch_1'):
                branch_1 = unit_3d.Unit3D(output_channels=112, kernel_shape = [1, 1, 1],
                    name = 'Conv3d_0z_1x1')(net, is_training = is_training)
                
                branch_1 = unit_3d.Unit3D(output_channels=224, kernel_shape = [3, 3, 3],
                    name = 'Conv3d_0b_3x3')(branch_1, is_training = is_training)

            with tf.compat.v1.variable_scope('Branch_2'):
                branch_2 = unit_3d.Unit3D(output_channels=24, kernel_shape = [1, 1, 1],
                    name = 'Conv3d_0a_1x1')(net, is_training = is_training)

                branch_2 = unit_3d.Unit3D(output_channels=64, kernel_shape = [3, 3, 3],
                    name = 'Conv3d_0b_3x3')(branch_2, is_training = is_training)
            
            with tf.compat.v1.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize = [1, 3, 3, 3, 1],
                    strides = [1, 1, 1, 1, 1], padding = snt.pad.same, name = 'MaxPool3d_0a_3x3')

                branch_3 = unit_3d.Unit3D(output_channels=64, kernel_shape = [1, 1, 1],
                    name = 'Conv3d_0b_1x1')(branch_3, is_training = is_training)

            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)

        end_points[end_point] = net

        if self._final_endpoint == end_point:
            return net, end_points

        end_point = 'Mixed_4d'
        with tf.compat.v1.variable_scope(end_point):
            with tf.compat.v1.variable_scope('Branch_0'):
                branch_0 = unit_3d.Unit3D(output_channels=128, kernel_shape = [1, 1, 1],
                    name = 'Conv3d_0a_1x1')(net, is_training = is_training)

            with tf.compat.v1.variable_scope('Branch_1'):
                branch_1 = unit_3d.Unit3D(output_channels=128, kernel_shape = [1, 1, 1],
                    name = 'Conv3d_0z_1x1')(net, is_training = is_training)
                
                branch_1 = unit_3d.Unit3D(output_channels=256, kernel_shape = [3, 3, 3],
                    name = 'Conv3d_0b_3x3')(branch_1, is_training = is_training)

            with tf.compat.v1.variable_scope('Branch_2'):
                branch_2 = unit_3d.Unit3D(output_channels=24, kernel_shape = [1, 1, 1],
                    name = 'Conv3d_0a_1x1')(net, is_training = is_training)

                branch_2 = unit_3d.Unit3D(output_channels=64, kernel_shape = [3, 3, 3],
                    name = 'Conv3d_0b_3x3')(branch_2, is_training = is_training)
            
            with tf.compat.v1.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize = [1, 3, 3, 3, 1],
                    strides = [1, 1, 1, 1, 1], padding = snt.pad.same, name = 'MaxPool3d_0a_3x3')

                branch_3 = unit_3d.Unit3D(output_channels=64, kernel_shape = [1, 1, 1],
                    name = 'Conv3d_0b_1x1')(branch_3, is_training = is_training)

            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)

        end_points[end_point] = net

        if self._final_endpoint == end_point:
            return net, end_points

        end_point = 'Mixed_4e'
        with tf.compat.v1.variable_scope(end_point):
            with tf.compat.v1.variable_scope('Branch_0'):
                branch_0 = unit_3d.Unit3D(output_channels=112, kernel_shape = [1, 1, 1],
                    name = 'Conv3d_0a_1x1')(net, is_training = is_training)

            with tf.compat.v1.variable_scope('Branch_1'):
                branch_1 = unit_3d.Unit3D(output_channels=144, kernel_shape = [1, 1, 1],
                    name = 'Conv3d_0z_1x1')(net, is_training = is_training)
                
                branch_1 = unit_3d.Unit3D(output_channels=288, kernel_shape = [3, 3, 3],
                    name = 'Conv3d_0b_3x3')(branch_1, is_training = is_training)

            with tf.compat.v1.variable_scope('Branch_2'):
                branch_2 = unit_3d.Unit3D(output_channels=32, kernel_shape = [1, 1, 1],
                    name = 'Conv3d_0a_1x1')(net, is_training = is_training)

                branch_2 = unit_3d.Unit3D(output_channels=64, kernel_shape = [3, 3, 3],
                    name = 'Conv3d_0b_3x3')(branch_2, is_training = is_training)
            
            with tf.compat.v1.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize = [1, 3, 3, 3, 1],
                    strides = [1, 1, 1, 1, 1], padding = snt.pad.same, name = 'MaxPool3d_0a_3x3')

                branch_3 = unit_3d.Unit3D(output_channels=64, kernel_shape = [1, 1, 1],
                    name = 'Conv3d_0b_1x1')(branch_3, is_training = is_training)

            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)

        end_points[end_point] = net

        if self._final_endpoint == end_point:
            return net, end_points

        end_point = 'Mixed_4f'
        with tf.compat.v1.variable_scope(end_point):
            with tf.compat.v1.variable_scope('Branch_0'):
                branch_0 = unit_3d.Unit3D(output_channels=256, kernel_shape = [1, 1, 1],
                    name = 'Conv3d_0a_1x1')(net, is_training = is_training)

            with tf.compat.v1.variable_scope('Branch_1'):
                branch_1 = unit_3d.Unit3D(output_channels=160, kernel_shape = [1, 1, 1],
                    name = 'Conv3d_0z_1x1')(net, is_training = is_training)
                
                branch_1 = unit_3d.Unit3D(output_channels=320, kernel_shape = [3, 3, 3],
                    name = 'Conv3d_0b_3x3')(branch_1, is_training = is_training)

            with tf.compat.v1.variable_scope('Branch_2'):
                branch_2 = unit_3d.Unit3D(output_channels=32, kernel_shape = [1, 1, 1],
                    name = 'Conv3d_0a_1x1')(net, is_training = is_training)

                branch_2 = unit_3d.Unit3D(output_channels=128, kernel_shape = [3, 3, 3],
                    name = 'Conv3d_0b_3x3')(branch_2, is_training = is_training)
            
            with tf.compat.v1.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize = [1, 3, 3, 3, 1],
                    strides = [1, 1, 1, 1, 1], padding = snt.pad.same, name = 'MaxPool3d_0a_3x3')

                branch_3 = unit_3d.Unit3D(output_channels=128, kernel_shape = [1, 1, 1],
                    name = 'Conv3d_0b_1x1')(branch_3, is_training = is_training)

            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)

        end_points[end_point] = net

        if self._final_endpoint == end_point:
            return net, end_points

        end_point = 'MaxPool3d_5a_2x2'
        net = tf.nn.max_pool3d(net, ksize = [1, 2, 2, 2, 1], strides = [1, 2, 2, 2, 1],
            padding = snt.pad.same, name = end_point)
        
        end_points[end_point] = net
        if self._final_endpoint == end_point:
            return net, end_points

        end_point = 'Mixed_5b'
        with tf.compat.v1.variable_scope(end_point):
            with tf.compat.v1.variable_scope('Branch_0'):
                branch_0 = unit_3d.Unit3D(output_channels=256, kernel_shape = [1, 1, 1],
                    name = 'Conv3d_0a_1x1')(net, is_training = is_training)

            with tf.compat.v1.variable_scope('Branch_1'):
                branch_1 = unit_3d.Unit3D(output_channels=160, kernel_shape = [1, 1, 1],
                    name = 'Conv3d_0z_1x1')(net, is_training = is_training)
                
                branch_1 = unit_3d.Unit3D(output_channels=320, kernel_shape = [3, 3, 3],
                    name = 'Conv3d_0b_3x3')(branch_1, is_training = is_training)

            with tf.compat.v1.variable_scope('Branch_2'):
                branch_2 = unit_3d.Unit3D(output_channels=32, kernel_shape = [1, 1, 1],
                    name = 'Conv3d_0a_1x1')(net, is_training = is_training)

                branch_2 = unit_3d.Unit3D(output_channels=128, kernel_shape = [3, 3, 3],
                    name = 'Conv3d_0b_3x3')(branch_2, is_training = is_training)
            
            with tf.compat.v1.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize = [1, 3, 3, 3, 1],
                    strides = [1, 1, 1, 1, 1], padding = snt.pad.same, name = 'MaxPool3d_0a_3x3')

                branch_3 = unit_3d.Unit3D(output_channels=128, kernel_shape = [1, 1, 1],
                    name = 'Conv3d_0b_1x1')(branch_3, is_training = is_training)

            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)

        end_points[end_point] = net

        if self._final_endpoint == end_point:
            return net, end_points

        end_point = 'Mixed_5c'
        with tf.compat.v1.variable_scope(end_point):
            with tf.compat.v1.variable_scope('Branch_0'):
                branch_0 = unit_3d.Unit3D(output_channels=384, kernel_shape = [1, 1, 1],
                    name = 'Conv3d_0a_1x1')(net, is_training = is_training)

            with tf.compat.v1.variable_scope('Branch_1'):
                branch_1 = unit_3d.Unit3D(output_channels=192, kernel_shape = [1, 1, 1],
                    name = 'Conv3d_0z_1x1')(net, is_training = is_training)
                
                branch_1 = unit_3d.Unit3D(output_channels=384, kernel_shape = [3, 3, 3],
                    name = 'Conv3d_0b_3x3')(branch_1, is_training = is_training)

            with tf.compat.v1.variable_scope('Branch_2'):
                branch_2 = unit_3d.Unit3D(output_channels=48, kernel_shape = [1, 1, 1],
                    name = 'Conv3d_0a_1x1')(net, is_training = is_training)

                branch_2 = unit_3d.Unit3D(output_channels=128, kernel_shape = [3, 3, 3],
                    name = 'Conv3d_0b_3x3')(branch_2, is_training = is_training)
            
            with tf.compat.v1.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize = [1, 3, 3, 3, 1],
                    strides = [1, 1, 1, 1, 1], padding = snt.pad.same, name = 'MaxPool3d_0a_3x3')

                branch_3 = unit_3d.Unit3D(output_channels=128, kernel_shape = [1, 1, 1],
                    name = 'Conv3d_0b_1x1')(branch_3, is_training = is_training)

            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)

        end_points[end_point] = net

        if self._final_endpoint == end_point:
            return net, end_points

        end_point = 'Logits'
        with tf.compat.v1.variable_scope(end_point):
            net = tf.nn.avg_pool3d(net, ksize = [1, 2, 7, 7, 1],
                strides = [1, 1, 1, 1, 1], padding=snt.pad.valid)

            net = tf.nn.dropout(net, dropout_keep_prob)
            logits = unit_3d.Unit3D(output_channels=self._num_classes, 
            kernel_shape=[1, 1, 1], activation_fn = None,
            use_batch_norm=False, use_bias = True,
            name = 'Conv3d_0c_1x1')(net, is_training = is_training)

            if self._spatial_squeeze:
                logits = tf.squeeze(logits, [2, 3], name = 'SpatialSqueeze')

            averaged_logits = tf.reduce_mean(logits, axis = 1)
            end_points[end_point] = averaged_logits
            if self._final_endpoint == end_point:
                return averaged_logits, end_points

            end_point = 'Predictions'
            predictions = tf.nn.softmax(averaged_logits)
            end_points[end_point] = predictions

            return predictions, end_points