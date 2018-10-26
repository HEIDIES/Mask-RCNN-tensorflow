import tensorflow as tf
import layer


class RPN:
    def __init__(self, name, anchors_per_location, anchor_stride,
                 is_training=True, use_bias=True):
        self.name = name
        self.anchors_per_location = anchors_per_location
        self.anchor_stride = anchor_stride
        self.is_training = is_training
        self.reuse = False
        self.use_bias = use_bias

    def __call__(self, ipt):

        with tf.variable_scope(self.name):
            opt = layer.rpn(ipt, anchors_per_location=self.anchors_per_location,
                            anchor_stride=self.anchor_stride, reuse=self.reuse,
                            is_training=self.is_training, use_bias=self.use_bias)

        self.reuse = True
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        return opt
