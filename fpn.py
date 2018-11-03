import ops
import tensorflow as tf


class FPN:
    def __init__(self, name, top_down_pyramid_size, use_bias):
        self.name = name
        self.use_bias = use_bias
        self.top_down_pyramid_size = top_down_pyramid_size
        self.reuse = False

    def __call__(self, ipt):
        c2, c3, c4, c5 = ipt
        with tf.variable_scope(self.name):
            p5 = ops.conv2d(c5, self.top_down_pyramid_size, 1, 0, 1, norm=None, activation=None,
                            reuse=self.reuse, kernel_initializer='glorot_uniform_tanh', use_bias=self.use_bias,
                            name='fpn_c5p5')
            p5_up = tf.tile(p5, multiples=[1, 2, 2, 1], name='fpn_p5upsampled')
            p4 = tf.add(p5_up, ops.conv2d(c4, self.top_down_pyramid_size, 1, 0, 1, norm=None, activation=None,
                                          reuse=self.reuse, kernel_initializer='glorot_uniform_tanh',
                                          use_bias=self.use_bias, name='fpn_c4p4'), name='fpn_p4add')
            p4_up = tf.tile(p4, multiples=[1, 2, 2, 1], name='fpn_p4upsampled')
            p3 = tf.add(p4_up, ops.conv2d(c3, self.top_down_pyramid_size, 1, 0, 1, norm=None, activation=None,
                                          reuse=self.reuse, kernel_initializer='glorot_uniform_tanh',
                                          use_bias=self.use_bias, name='fpn_c3p3'), name='fpn_p3add')
            p3_up = tf.tile(p3, multiples=[1, 2, 2, 1], name='fpn_p3upsampled')
            p2 = tf.add(p3_up, ops.conv2d(c2, self.top_down_pyramid_size, 1, 0, 1, norm=None, activation=None,
                                          reuse=self.reuse, kernel_initializer='glorot_uniform_tanh',
                                          use_bias=self.use_bias, name='fpn_c2p2'), name='fpn_p2add')
            p2 = ops.conv2d(p2, self.top_down_pyramid_size, 3, 1, 1, norm=None, activation=None, reuse=self.reuse,
                            kernel_initializer='glorot_uniform_tanh', use_bias=self.use_bias, name='fpn_p2')
            p3 = ops.conv2d(p3, self.top_down_pyramid_size, 3, 1, 1, norm=None, activation=None, reuse=self.reuse,
                            kernel_initializer='glorot_uniform_tanh', use_bias=self.use_bias, name='fpn_p3')
            p4 = ops.conv2d(p4, self.top_down_pyramid_size, 3, 1, 1, norm=None, activation=None, reuse=self.reuse,
                            kernel_initializer='glorot_uniform_tanh', use_bias=self.use_bias, name='fpn_p4')
            p5 = ops.conv2d(p5, self.top_down_pyramid_size, 3, 1, 1, norm=None, activation=None, reuse=self.reuse,
                            kernel_initializer='glorot_uniform_tanh', use_bias=self.use_bias, name='fpn_p5')
            p6 = tf.nn.max_pool(p5, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name='fpn_p6')

        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.reuse = True
        return [p2, p3, p4, p5, p6]