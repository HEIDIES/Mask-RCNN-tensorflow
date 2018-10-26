import layer
import tensorflow as tf


class FpnClassifier:
    def __init__(self, name, pool_size, num_classes, is_training=False):
        self.name = name
        self.pool_size = pool_size
        self.num_classes = num_classes
        self.is_training = is_training

    def __call__(self, ipt):
        with tf.variable_scope(self.name):
            opt = layer.fpn_classifier(ipt, pool_size=self.pool_size, num_classes=self.num_classes,
                                       is_training=self.is_training)

        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        return opt
