'''
Inception-v1 Inflated 3D convnet used for Kinetics CVPR paper.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonnet as snt
import tensorflow as tf

class Unit3D(snt.Module):
    '''Basic unit containing Conv3D + BatchNorm  + Non-linearity'''

    def __init__(self, output_channels, 
    kernel_shape = (1, 1, 1),
    stride = (1, 1, 1),
    activation_fn = tf.nn.relu,
    use_batch_norm = True,
    use_bias = False,
    name = 'unit_3d'):
        #Initializes Unit3D module.
        super(Unit3D, self).__init__(name = name)
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias

    def build(self, inputs, is_training):
        '''Connects the module to inputs.
        
        Args:
            inputs: Inputs to the Unit3D component.
            is_training: Whether to use the training mode for snt.BatchNorm (boolean).
            
        Returns:
            Outputs from the module.
        '''

        net = snt.Conv3D(output_channels=self._output_channels,
        kernel_shape = self._kernel_shape,
        stride = self._stride,
        padding = snt.pad.same,
        use_bias = self._use_bias)(inputs)

        if self._use_batch_norm:
            bn = snt.BatchNorm()
            net = bn(net, is_training = is_training, test_local_stats = False)
        
        if self._activation_fn is not None:
            net = self._activation_fn(net)
        return net
