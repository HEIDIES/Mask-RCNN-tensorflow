import tensorflow as tf
import ops
import utils


class PyramidRoiPooling:
    def __init__(self, name, roi_size=7):
        self.name = name
        self.roi_size = roi_size

    def __call__(self, ipt):
        boxes = ipt[0]
        image_meta = ipt[1]
        feature_maps = ipt[2]

        y1, x1, y2, x2 = tf.split(boxes, [1, 1, 1, 1], axis=2)

        w = x2 - x1
        h = y2 - y1

        image_shape = utils.parse_image_meta_graph(image_meta)['image_shape'][0]
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)

        roi_level = ops.log2_graph(tf.sqrt(w * h) / (224.0 / tf.sqrt(image_area)))
        roi_level = tf.minimum(5, tf.maximum(
            2, 4 + tf.cast(tf.round(roi_level), tf.int32)))

        roi_level = tf.squeeze(roi_level, 2)

        pooled = []
        box_to_level = []

        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)

            # Box indices for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so
            # that we can evaluate either max or average pooling. In fact,
            # interpolating only a single value at each bin center (without
            # pooling) is nearly as effective."
            #
            # Here we use the simplified approach of a single value per bin,
            # which is how it's done in tf.crop_and_resize()
            # Result: [batch * num_boxes, pool_height, pool_width, channels]
            pooled.append(tf.image.crop_and_resize(
                feature_maps[i], level_boxes, box_indices, [self.roi_size, self.roi_size],
                method="bilinear"))

        pooled = tf.concat(pooled, axis=0)

        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)

        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
            box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
        pooled = tf.reshape(pooled, shape)
        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + [self.roi_size, self.roi_size] + (input_shape[2][-1], )
