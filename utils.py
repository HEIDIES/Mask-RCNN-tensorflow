import tensorflow as tf
import random


def convert2int(image):
    """ Transfrom from float tensor ([-1.,1.]) to int image ([0,255])
    """
    return tf.image.convert_image_dtype((image+1.0)/2.0, tf.uint8)


def convert2float(image):
    """ Transfrom from int image ([0,255]) to float tensor ([-1.,1.])
    """
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return (image/127.5) - 1.0


def batch_convert2int(images):
    """
    Args:
      images: 4D float tensor (batch_size, image_size, image_size, depth)
    Returns:
      4D int tensor
    """
    return tf.map_fn(convert2int, images, dtype=tf.uint8)


def batch_convert2float(images):
    """
    Args:
      images: 4D int tensor (batch_size, image_size, image_size, depth)
    Returns:
      4D float tensor
    """
    return tf.map_fn(convert2float, images, dtype=tf.float32)


class ImagePool:
    """ History of generated images
        Same logic as https://github.com/junyanz/CycleGAN/blob/master/util/image_pool.lua
    """
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.images = []

    def query(self, image):
        if self.pool_size == 0:
            return image

        if len(self.images) < self.pool_size:
            self.images.append(image)
            return image
        else:
            p = random.random()
            if p > 0.5:
                # use old image
                random_id = random.randrange(0, self.pool_size)
                tmp = self.images[random_id].copy()
                self.images[random_id] = image.copy()
                return tmp
            else:
                return image


def get_anchors(anchors_path):
    """
    Reads the anchors from a file and returns them in a list.

    Parameters
    ----------
    anchors_path : string
        Path that points to where the anchor information is stored.

    Returns
    -------
    anchors : list
        A list of format:
        [[anchor1_width, anchor1_height], [anchor2_width, anchor2_height], [anchor3_width, anchor3_height], ...]
    """
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = list(zip(anchors[::2], anchors[1::2]))

    return anchors


def batch_slice(ipt, tf_fn, batch_size, names=None):
    """
    Args:
        ipt: input tensor, or a list of input tensor
        tf_fn: some tensorflow function with tensor output can only support a batch size of 1 only.
        batch_size: number of slices to divide the data into.
        names: If provided, assigns names to the resulting tensors.

    Returns:
        opt: output tensor, or a list of output tensor
    """
    if isinstance(ipt, list):
        ipt = [ipt]

    opt = []

    for i in range(batch_size):
        ipt_slice = [x[i] for x in ipt]
        opt_slice = tf_fn(*ipt_slice)

        if isinstance(opt_slice, (tuple, list)):
            opt_slice = [opt_slice]

        opt.append(opt_slice)
    opt = list(zip(*opt))

    if names is None:
        names = [None] * len(opt)

    result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(opt, names)]
    if len(result) == 1:
        result = result[0]

    return result


def parse_image_meta_graph(meta):
    """
    Args:
        meta: [batch, meta length] where meta length depends on NUM_CLASSES

    Returns:
        Returns a dict of the parsed tensors.
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {
        "image_id": image_id,
        "original_image_shape": original_image_shape,
        "image_shape": image_shape,
        "window": window,
        "scale": scale,
        "active_class_ids": active_class_ids,
    }


def trim_zeros_graph(boxes, name=None):
    """
    Args:
        boxes: [N, 4] matrix of boxes
        name: name of the operator if needed

    Returns:
        [N, 4] matrix of boxes
        [N] a 1D boolean mask identifying the rows to keep
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


def boxes_to_delta(boxes, gt_boxes):
    """
    Args:
        boxes: [N, (y1, x1, y2, x2)]
        gt_boxes: [N, (y1, x1, y2, x2)]

    Returns:
        [N, (dy, dx, log(dh), log(dw))]
    """
    b_y1, b_x1, b_y2, b_x2 = tf.split(boxes, 4, axis=0)
    gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(gt_boxes, 4, axis=0)

    b_h = b_y2 - b_y1
    b_w = b_x2 - b_x1
    gt_h = gt_y2 - gt_y1
    gt_w = gt_x2 - gt_x1

    b_y = b_y1 + 0.5 * b_h
    b_x = b_x1 + 0.5 * b_w
    gt_y = gt_y1 + 0.5 * gt_h
    gt_x = gt_x1 + 0.5 * gt_w

    dy = (gt_y - b_y) / b_h
    dx = (gt_x - b_x) / b_w

    dh = gt_h / b_h
    dw = gt_w / b_w
    dh = tf.log(dh)
    dw = tf.log(dw)

    return tf.concat([dy, dx, dh, dw], axis=0)


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
