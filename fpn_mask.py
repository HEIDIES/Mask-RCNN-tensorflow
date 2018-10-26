import layer
import tensorflow as tf


class FpnMask:
    def __init__(self, name, num_classes, is_training=False):
        self.name = name
        self.num_classes = num_classes
        self.is_training = is_training

    def __call__(self, ipt):
        with tf.variable_scope(self.name):
            opt = layer.build_fpn_mask(ipt, num_classes=self.num_classes, is_training=self.is_training)

        self.vat_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        return opt
