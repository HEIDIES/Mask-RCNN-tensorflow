import tensorflow as tf
import layer
import ops


class Resnet:
    def __init__(self, name, architecture, is_training, stage5=False, norm='batch', use_bias=True):
        self.name = name
        self.architecture = architecture
        self.stage5 = stage5
        self.is_training = is_training
        self.norm = norm
        self.reuse = False
        self.use_bias = use_bias

    def __call__(self, ipt):

        assert self.architecture in ['resnet50', 'resnet101']

        with tf.variable_scope(self.name):

            # stage1

            c7s2k64 = ops.conv2d(ipt, 64, 7, 3, 2, name='conv1', reuse=self.reuse, is_training=self.is_training,
                                 norm=self.norm, use_bias=self.use_bias, activation=tf.nn.relu)

            c1 = c7s2k64 = tf.nn.max_pool(c7s2k64, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')

            # stage2

            conv2_out = layer.conv_block(c7s2k64, [64, 64, 256], stage=2, block='a',
                                         norm=self.norm, reuse=self.reuse,
                                         is_training=self.is_training, use_bias=self.use_bias)

            id2_out = layer.identity_block(conv2_out, [64, 64, 256], stage=2, block='b',
                                           use_bias=self.use_bias, norm=self.norm, reuse=self.reuse,
                                           is_training=self.is_training)

            c2 = id2_out = layer.identity_block(id2_out, [64, 64, 256], stage=2, block='c',
                                                use_bias=self.use_bias, norm=self.norm, reuse=self.reuse,
                                                is_training=self.is_training)

            # stage3

            conv3_out = layer.conv_block(id2_out, [128, 128, 512], stage=3, block='a',
                                         norm=self.norm, reuse=self.reuse,
                                         is_training=self.is_training, use_bias=self.use_bias)

            id3_out = layer.identity_block(conv3_out, [128, 128, 512], stage=3, block='b',
                                           norm=self.norm, reuse=self.reuse,
                                           is_training=self.is_training, use_bias=self.use_bias)

            id3_out = layer.identity_block(id3_out, [128, 128, 512], stage=3, block='c',
                                           norm=self.norm, reuse=self.reuse,
                                           is_training=self.is_training, use_bias=self.use_bias)

            c3 = id3_out = layer.identity_block(id3_out, [128, 128, 512], stage=3, block='d',
                                                norm=self.norm, reuse=self.reuse,
                                                is_training=self.is_training, use_bias=self.use_bias)

            # stage4

            conv4_out = layer.conv_block(id3_out, [256, 256, 1024], stage=4, block='a',
                                         norm=self.norm, reuse=self.reuse,
                                         is_training=self.is_training, use_bias=self.use_bias)

            block_count = {"resnet50": 5, "resnet101": 22}[self.architecture]
            id4_out = conv4_out
            for i in range(block_count):
                id4_out = layer.identity_block(id4_out, [256, 256, 1024], stage=4, block=chr(98 + i),
                                               norm=self.norm, reuse=self.reuse,
                                               is_training=self.is_training, use_bias=self.use_bias)
            c4 = id4_out

            # Stage5

            if self.stage5:
                conv5_out = layer.conv_block(id4_out, [512, 512, 2048], stage=5, block='a',
                                             norm=self.norm, reuse=self.reuse,
                                             is_training=self.is_training, use_bias=self.use_bias)

                id5_out = layer.identity_block(conv5_out, [512, 512, 2048], stage=5, block='b',
                                               norm=self.norm, reuse=self.reuse,
                                               is_training=self.is_training, use_bias=self.use_bias)
                c5 = layer.identity_block(id5_out, [512, 512, 2048], stage=5, block='c',
                                          norm=self.norm, reuse=self.reuse,
                                          is_training=self.is_training, use_bias=self.use_bias)
            else:
                c5 = None

        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.reuse = True
        return [c1, c2, c3, c4, c5]
