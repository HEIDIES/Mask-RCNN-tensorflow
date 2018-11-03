import tensorflow as tf


def leaky_relu(x):
    """A implementation of leaky_relu function.
    Args:
        x: 2-D or 4-D tensor, e.g. [batch_size, num_dims], [batch_size, height, width, channels]
    Returns:
        2-D or 4-D tensor, activation value of input.
    """
    return tf.where(tf.greater(x, 0), x, 0.001 * x)


def _weights(name, shape, mean=0.0, stddev=0.01, initializer=None):
    """Weights initializer.
    Args:
        name: string, the name of weight, e.g. weights
        shape: list, the shape of weight. e.g. [input_dims, output_dims]
        mean: float, the mean of the weight, default: 0.0
        stddev: float, the standard deviation, default: 0.02
        initializer: string, weights initializer methods, default: None
    Returns:
        var: x-D tensor, weights.
    """
    if initializer == 'glorot_normal_tanh':
        fin = shape[2]
        fout = shape[3]
        if len(shape) == 2:
            fin = shape[0]
            fout = shape[1]
        fin = tf.cast(fin, tf.float32)
        fout = tf.cast(fout, tf.float32)
        var = tf.get_variable(
            name, shape, initializer=tf.random_normal_initializer(
                mean=mean, stddev=tf.sqrt(2. / (fin + fout)), dtype=tf.float32
            )
        )
    elif initializer == 'glorot_uniform_tanh':
        var = tf.get_variable(
            name, shape, initializer=tf.contrib.layers.xavier_initializer()
        )
    elif initializer == 'glorot_normal_sigmoid':
        fin = shape[2]
        fout = shape[3]
        if len(shape) == 2:
            fin = shape[0]
            fout = shape[1]
        fin = tf.cast(fin, tf.float32)
        fout = tf.cast(fout, tf.float32)
        var = tf.get_variable(
            name, shape, initializer=tf.random_normal_initializer(
                mean=mean, stddev=4 * tf.sqrt(2. / (fin + fout)), dtype=tf.float32
            )
        )
    elif initializer == 'glorot_uniform_sigmoid':
        fin = shape[2]
        fout = shape[3]
        if len(shape) == 2:
            fin = shape[0]
            fout = shape[1]
        fin = tf.cast(fin, tf.float32)
        fout = tf.cast(fout, tf.float32)
        var = tf.get_variable(
            name, shape, initializer=tf.random_uniform_initializer(
                minval=-4 * tf.sqrt(6. / (fin + fout)), dtype=tf.float32
            )
        )
    elif initializer == 'he_normal':
        fin = shape[2]
        if len(shape) == 2:
            fin = shape[0]
        fin = tf.cast(fin, tf.float32)
        var = tf.get_variable(
            name, shape, initializer=tf.random_normal_initializer(
                mean=mean, stddev=tf.sqrt(2. / fin), dtype=tf.float32
            )
        )
    elif initializer == 'he_uniform':
        fin = shape[2]
        if len(shape) == 2:
            fin = shape[0]
        fin = tf.cast(fin, tf.float32)
        var = tf.get_variable(
            name, shape, initializer=tf.random_normal_initializer(
                mean=mean, stddev=tf.sqrt(6. / fin), dtype=tf.float32
            )
        )
    else:
        var = tf.get_variable(
            name, shape,
            initializer=tf.random_normal_initializer(
                mean=mean, stddev=stddev, dtype=tf.float32
            )
        )
    return var


def _bias(name, shape, constant=0.0):
    """
    :param name: string, the name of bias, e.g. 'bias'
    :param shape: list, the shape of bias.
    :param constant: float, the initial value of bias
    :return: bias
    """
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(constant))


def fully_connected(x, output_dims, use_bias=True, is_training=True, reuse=False,
                    name=None, activation=None, norm=None, weights_initializer=None):
    """
    :param x: 2D tensor.
    :param output_dims: int, the number of output dimensions.
    :param use_bias: bool, use bias or not
    :param is_training: bool, is training or not
    :param reuse: bool, reuse or not
    :param name: string, name of the fully_connected layer
    :param activation: function, the activation function
    :param norm: string, use norm or not, and what kind of norm methods will be used
    :param weights_initializer: string, determine what kind of initializer methods of weights will be used.
    :return: 2D tensor, output of fully_connected layer
    """
    with tf.variable_scope(name, reuse=reuse):
        weights = _weights(name='weights', shape=[x.get_shape()[1], output_dims],
                           initializer=weights_initializer)
        x = tf.matmul(x, weights)
        if use_bias:
            bias = _bias('bias', [output_dims])
            x = tf.add(x, bias)
        if norm is not None:
            x = _norm(x, norm, is_training)
        if activation is not None:
            x = activation(x)
        return x


def conv2d(x, filters, ksize, pad_size=0, stride=1, pad_mode='CONSTANT', padding='VALID',
           norm=None, activation=None, name='conv2d', reuse=False, is_training=True,
           kernel_initializer=None, use_bias=False, upsampling=None, act_first=False,
           weights_std=0.001, bias_init=0.0):
    """
    :param x: 4D tensor, [batch_size, height, width, channels]
    :param filters: int, number of output channels
    :param ksize: int, size of convolution kernel
    :param pad_size: int, size of padding
    :param stride: int, stride of convolution kernel
    :param pad_mode: string, the method of padding
    :param padding: string, another way to set the padding method
    :param norm: string, use norm or not, and what kind of norm methods will be used
    :param activation: function, the activation function
    :param name: string, name of convolution layer
    :param reuse: bool, use bias or not
    :param is_training: bool, is training or not
    :param kernel_initializer:  string, determine what kind of initializer methods of weights will be used.
    :param use_bias: bool, use bias or not
    :param upsampling: list, copy the input upsampling[0] times by row, and upsampling[1] times by column.
    :param act_first: bool, activate before or after BN
    :param weights_std: float, initial std of weights
    :param bias_init: float, initial value of bias
    :return: 4D tensor, output of convolution layer
    """
    with tf.variable_scope(name, reuse=reuse):
        if upsampling is not None:
            x = tf.tile(x, multiples=[1, upsampling[0], upsampling[1], 1])
        input_shape = x.get_shape()[3]
        weights = _weights('weights', shape=[ksize, ksize, input_shape, filters], initializer=kernel_initializer,
                           stddev=weights_std)
        if pad_size > 0:
            x = tf.pad(x, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], mode=pad_mode)
        x = tf.nn.conv2d(
            x, weights,
            strides=[1, stride, stride, 1], padding=padding,
            name=name
        )
        if use_bias:
            bias = _bias('bias', [filters], constant=bias_init)
            x = tf.add(x, bias)
        if act_first:
            if activation is not None:
                x = activation(x)
            if norm is not None:
                x = _norm(x, norm, is_training)
        else:
            if norm is not None:
                x = _norm(x, norm, is_training)
            if activation is not None:
                x = activation(x)
        return x


def unconv2d(x, output_dims, ksize, stride=1, norm=None, activation=None,
             name=None, reuse=False, use_bias=False, is_training=True, kernel_initializer=None):
    """
    :param x: 4D tensor, [batch_size, height, width, channels]
    :param output_dims: int, output_dims
    :param ksize: int, size of kernel
    :param stride: stride of kernel
    :param norm: string, use norm or not, and what kind of norm methods will be used
    :param activation: function, the activation function
    :param name: string, name of convolution layer
    :param reuse: bool, use bias or not
    :param use_bias: bool, use bias or not
    :param is_training: bool, is training or not
    :param kernel_initializer: string, determine what kind of initializer methods of weights will be used.
    :return: 4D tensor, output of conv_transpose layer
    """
    input_shape = x.get_shape().as_list()
    with tf.variable_scope(name, reuse=reuse):
        weights = _weights(name='weights', shape=[ksize, ksize, output_dims, input_shape[3]],
                           initializer=kernel_initializer)
        x = tf.nn.conv2d_transpose(
            x, weights, [tf.shape(x)[0], input_shape[1] * stride,
                         input_shape[2] * stride, output_dims],
            strides=[1, stride, stride, 1], padding='SAME'
        )
        if use_bias:
            bias = _bias('bias', [output_dims])
            x = tf.add(x, bias)
        if norm is not None:
            x = _norm(x, norm, is_training)
        if activation is not None:
            x = activation(x)
        return x


def _norm(x, norm, is_training, activation=None):
    """
    :param x: 2D or 4D tensor.
    :param norm: string, norm method
    :param is_training: bool, is training or not
    :param activation: function, the activation function
    :return:
    """
    if norm == 'batch':
        return _batch_norm(x, is_training, activation=activation)
    if norm == 'instance':
        return _instance_norm(x)


def _batch_norm(x, is_training, activation=None):
    """
    :param x: 2D or 4D tensor
    :param is_training: bool, is training or not
    :param activation: function, the activation function
    :return: batch normalization of input
    """
    with tf.variable_scope('batch_normalization'):
        x = tf.contrib.layers.batch_norm(x,
                                         decay=0.9,
                                         scale=True,
                                         updates_collections=None,
                                         is_training=is_training)
        if activation is not None:
            x = activation(x)
        return x


def _instance_norm(x, activation=None):
    """
    :param x: 2D or 4D tensor
    :param activation: function, the activation function
    :return: instance normalization of input
    """
    with tf.variable_scope('instance_norm'):
        depth = x.get_shape()[3]
        scale = _weights('scale', [depth], mean=1.0)
        offset = _bias('offset', [depth])
        axis = [1, 2]
        mean, var = tf.nn.moments(x, axis, keep_dims=True)
        inv = tf.rsqrt(var + 1e-5)
        x = scale * (x - mean) * inv + offset
        if activation is not None:
            x = activation(x)
        return x


def max_pool_with_argmax(x, stride=2):
    """
    :param x: 4D tensor
    :param stride: int, stride of pool kernel
    :return: 4D tensor, output of pool layer
    """
    with tf.variable_scope('maxpooling'):
        _, mask = tf.nn.max_pool_with_argmax(x, ksize=[1, stride, stride, 1],
                                             strides=[1, stride, stride, 1], padding='SAME')
        mask = tf.stop_gradient(mask)
        x = tf.nn.max_pool(x, ksize=[1, stride, stride, 1],
                           strides=[1, stride, stride, 1], padding='SAME')
        return x, mask


def unpool(x, mask, stride=2):
    """
    :param x: 4D tensor.
    :param mask: list, location of the activated value
    :param stride: stride of unpool layer
    :return: 4D tensor, output of unpool layer
    """
    with tf.variable_scope('unpooling'):
        ksize = [1, stride, stride, 1]
        input_shape = x.get_shape().as_list()

        output_shape = (input_shape[0], input_shape[1] * ksize[1],
                        input_shape[2] * ksize[2], input_shape[3])

        one_like_mask = tf.ones_like(mask)
        batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int64),
                                 shape=[input_shape[0], 1, 1, 1])

        b = one_like_mask * batch_range
        y = mask // (output_shape[2] * output_shape[3])
        x = mask % (output_shape[2] * output_shape[3]) // output_shape[3]
        feature_range = tf.range(output_shape[3], dtype=tf.int64)
        f = one_like_mask * feature_range

        updates_size = tf.size(x)
        indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
        values = tf.reshape(x, [updates_size])
        ret = tf.scatter_nd(indices, values, output_shape)
        return ret


def global_avg_pool(x):
    """
    :param x: 4D tensor
    :return: float, instance mean of the input
    """
    with tf.variable_scope('global_avg_pooling'):
        input_shape = x.get_shape().as_list()
        return tf.nn.avg_pool(x, ksize=[1, input_shape[1], input_shape[2], 1],
                              strides=[1, input_shape[1], input_shape[2], 1], padding='SAME')


def log2_graph(x):
    """
    :param x: 4D tensor
    :return: float, log2 of input
    """
    return tf.log(x) / tf.log(2.0)


def iou(boxes1, boxes2):
    """
    :param boxes1: 4D tensor, [N, (y1, x1, y2, x2)]
    :param boxes2: 4D tensor, [N, (y1, x1, y2, x2)]
    :return: iou of two boxes
    """
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1), [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])

    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, [1, 1, 1, 1], axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, [1, 1, 1, 1], axis=1)

    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)

    intersections = (y2 - y1) * (x2 - x1)

    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)

    union = b1_area + b2_area - intersections

    overlaps = intersections / union

    overlaps = tf.reshape(overlaps, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])

    return overlaps


class BatchNorm(tf.keras.layers.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """
    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)


def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typically: [N, 4], but could be any shape.
    """
    diff = tf.abs(y_true - y_pred)
    less_than_one = tf.cast(tf.less(diff, 1.0), tf.float32)
    loss = (less_than_one * 0.5 * tf.square(diff)) + (1 - less_than_one) * (diff - tf.constant(0.5))
    return loss


def batch_pack_graph(x, counts, num_rows):
    """Picks different number of values from each row
    in x depending on the values in counts.
    """
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)


def norm_boxes(boxes, shape):
    """
    Args:
        boxes: [..., (y1, x1, y2, x2)] in pixel coordinates
        shape: [..., (height, width)] in pixels

    Returns:
        [..., (y1, x1, y2, x2)] in normalized coordinates
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2, axis=-1)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.divide(boxes - shift, scale)
