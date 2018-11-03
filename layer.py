import tensorflow as tf
import ops
import utils
import hyper_parameters
import numpy as np


def identity_block(ipt, filters, stage, block, use_bias=True, norm='batch', reuse=False, is_training=True):

    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'

    c1s1kx = ops.conv2d(ipt, nb_filter1, 1, 0, 1, norm=norm, activation=tf.nn.relu,
                        name=conv_name_base + '2a', reuse=reuse, is_training=is_training,
                        use_bias=use_bias, kernel_initializer='glorot_uniform_tanh')

    c3s1kx = ops.conv2d(c1s1kx, nb_filter2, 3, 1, 1, norm=norm, activation=tf.nn.relu,
                        name=conv_name_base + '2b', reuse=reuse, is_training=is_training,
                        use_bias=use_bias, kernel_initializer='glorot_uniform_tanh')

    c1s1kx = ops.conv2d(c3s1kx, nb_filter3, 1, 0, 1, norm=norm, activation=None,
                        name=conv_name_base + '2c', reuse=reuse, is_training=is_training,
                        use_bias=use_bias, kernel_initializer='glorot_uniform_tanh')

    opt = tf.add(c1s1kx, ipt)
    opt = tf.nn.relu(opt, name=str(stage) + block + '_out')

    return opt


def conv_block(ipt, filters, stage, block, norm='batch', reuse=False, is_training=True, use_bias=True):

    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'

    c1s2kx = ops.conv2d(ipt, nb_filter1, 1, 0, 2, norm=norm, activation=tf.nn.relu,
                        name=conv_name_base + '2a', reuse=reuse,
                        is_training=is_training, use_bias=use_bias, kernel_initializer='glorot_uniform_tanh')

    c3s1kx = ops.conv2d(c1s2kx, nb_filter2, 3, 1, 1, norm=norm, activation=tf.nn.relu,
                        name=conv_name_base + '2b', reuse=reuse, is_training=is_training,
                        use_bias=use_bias, kernel_initializer='glorot_uniform_tanh')

    c1s1kx = ops.conv2d(c3s1kx, nb_filter3, 1, 0, 1, norm=norm, activation=None,
                        name=conv_name_base + '2c', reuse=reuse, is_training=is_training,
                        use_bias=use_bias, kernel_initializer='glorot_uniform_tanh')

    shortcut = ops.conv2d(ipt, nb_filter3, 1, 0, 2, norm=norm, activation=None,
                          name=conv_name_base + '1', reuse=reuse, is_training=is_training,
                          use_bias=use_bias, kernel_initializer='glorot_uniform_tanh')

    opt = tf.add(c1s1kx, shortcut)

    opt = tf.nn.relu(opt, name=str(stage) + block + '_out')

    return opt


def apply_box_deltas_graph(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, (y1, x1, y2, x2)] boxes to update
    deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    return result


def clip_boxes_graph(boxes, window):
    """
    boxes: [N, (y1, x1, y2, x2)]
    window: [4] in the form y1, x1, y2, x2
    """
    # Split
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    clipped.set_shape((clipped.shape[0], 4))
    return clipped


def target_detection(proposals, gt_class_ids, gt_boxes, gt_masks):

    # assert_ = tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals], name='roi_assert')

    # with tf.control_dependencies(assert_):
    #     proposals = tf.identity(proposals)

    proposals, _ = utils.trim_zeros_graph(proposals, name="trim_proposals")
    gt_boxes, non_zeros = utils.trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros,
                                   name="trim_gt_class_ids")
    gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=2,
                         name="trim_gt_masks")

    crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
    non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
    crowd_boxes = tf.gather(gt_boxes, crowd_ix)
    # crowd_masks = tf.gather(gt_masks, crowd_ix, axis=2)
    gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
    gt_boxes = tf.gather(gt_boxes, non_crowd_ix)
    gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=2)

    ious = ops.iou(proposals, gt_boxes)

    crowd_ious = ops.iou(proposals, crowd_boxes)
    crowd_iou_max = tf.reduce_max(crowd_ious, axis=1)
    no_crowd_bool = (crowd_iou_max < 0.001)

    roi_iou_max = tf.reduce_max(ious, axis=1)

    positive_roi_bool = (roi_iou_max >= 0.5)
    positive_indices = tf.where(positive_roi_bool)[:, 0]

    negative_indices = tf.where(tf.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]

    positive_count = int(hyper_parameters.FLAGS.TRAIN_ROIS_PER_IMAGE *
                         hyper_parameters.FLAGS.ROI_POSITIVE_RATIO)
    positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
    positive_count = tf.shape(positive_indices)[0]

    r = 1.0 / hyper_parameters.FLAGS.ROI_POSITIVE_RATIO
    negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    negative_indices = tf.random_shuffle(negative_indices)[:negative_count]

    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)

    positive_ious = tf.gather(ious, positive_indices)
    roi_gt_box_assignment = tf.cond(
        tf.greater(tf.shape(positive_ious)[1], 0),
        true_fn=lambda: tf.argmax(positive_ious, axis=1),
        false_fn=lambda: tf.cast(tf.constant([]), tf.int64)
    )

    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

    deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
    deltas /= np.array(hyper_parameters.FLAGS.BBOX_STD_DEV)

    transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)
    roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)

    boxes = positive_rois
    if hyper_parameters.FLAGS.USE_MINI_MASK:
        y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
        gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
        gt_h = gt_y2 - gt_y1
        gt_w = gt_x2 - gt_x1
        y1 = (y1 - gt_y1) / gt_h
        x1 = (x1 - gt_x1) / gt_w
        y2 = (y2 - gt_y1) / gt_h
        x2 = (x2 - gt_x1) / gt_w
        boxes = tf.concat([y1, x1, y2, x2], 1)

    box_ids = tf.range(0, tf.shape(roi_masks)[0])
    masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes,
                                     box_ids,
                                     hyper_parameters.FLAGS.MASK_SHAPE)

    masks = tf.squeeze(masks, axis=3)
    masks = tf.round(masks)

    rois = tf.concat([positive_rois, negative_rois], axis=0)
    n = tf.shape(negative_rois)[0]
    p = tf.maximum(hyper_parameters.FLAGS.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
    rois = tf.pad(rois, [(0, p), (0, 0)])
    # roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, n + p), (0, 0)])
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, n + p)])
    deltas = tf.pad(deltas, [(0, n + p), (0, 0)])
    masks = tf.pad(masks, [[0, n + p], (0, 0), (0, 0)])

    return rois, roi_gt_class_ids, deltas, masks


def refine_detections(rois, probs, deltas, window):
    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)

    indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
    class_scores = tf.gather_nd(probs, indices)

    deltas_specific = tf.gather_nd(deltas, indices)

    refined_rois = apply_box_deltas_graph(
        rois, deltas_specific * hyper_parameters.FLAGS.BBOX_STD_DEV)

    refined_rois = clip_boxes_graph(refined_rois, window)

    keep = tf.where(class_ids > 0)[:, 0]
    if hyper_parameters.FLAGS.DETECTION_MIN_CONFIDENCE:
        conf_keep = tf.where(class_scores >= hyper_parameters.FLAGS.DETECTION_MIN_CONFIDENCE)[:, 0]
        keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                        tf.expand_dims(conf_keep, 0))
        keep = tf.sparse_tensor_to_dense(keep)[0]

    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(class_scores, keep)
    pre_nms_rois = tf.gather(refined_rois, keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    def nms_keep_map(class_id):
        ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
        class_keep = tf.image.non_max_suppression(
            tf.gather(pre_nms_rois, ixs),
            tf.gather(pre_nms_scores, ixs),
            max_output_size=hyper_parameters.FLAGS.DETECTION_MAX_INSTANCES,
            iou_threshold=hyper_parameters.FLAGS.DETECTION_NMS_THRESHOLD)

        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
        gap = hyper_parameters.FLAGS.DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0]
        class_keep = tf.pad(class_keep, [(0, gap)],
                            mode='CONSTANT', constant_values=-1)

        class_keep.set_shape([hyper_parameters.FLAGS.DETECTION_MAX_INSTANCES])
        return class_keep

    nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids,
                         dtype=tf.int64)

    nms_keep = tf.reshape(nms_keep, [-1])
    nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])

    keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                    tf.expand_dims(nms_keep, 0))
    keep = tf.sparse_tensor_to_dense(keep)[0]

    class_scores_keep = tf.gather(class_scores, keep)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0],
                          hyper_parameters.FLAGS.DETECTION_MAX_INSTANCES)

    top_k_indices = tf.nn.top_k(keep, num_keep).indices

    keep = tf.gather(keep, top_k_indices)

    detections = tf.concat([
        tf.gather(refined_rois, keep),
        tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis],
        tf.gather(class_scores, keep)[..., tf.newaxis]
    ], axis=1)

    gap_ = hyper_parameters.FLAGS.DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
    detections = tf.pad(detections, [(0, gap_), (0, 0)], "CONSTANT")
    return detections


def rpn(ipt, anchors_per_location, anchor_stride, reuse=False, is_training=True, use_bias=True):
    c3s1k512 = ops.conv2d(ipt, 512, 3, 1, anchor_stride, norm=None, activation=tf.nn.relu,
                          reuse=reuse, is_training=is_training, name='rpn_conv_shared',
                          use_bias=use_bias, kernel_initializer='glorot_uniform_tanh')

    c1s1k2x = ops.conv2d(c3s1k512, 2 * anchors_per_location, 1, 0, 1, norm=None, activation=None,
                         reuse=reuse, is_training=is_training, name='rpn_class_raw',
                         use_bias=use_bias, kernel_initializer='glorot_uniform_tanh')

    rpn_class_logits = tf.reshape(c1s1k2x, [tf.shape(c1s1k2x)[0], -1, 2])

    rpn_probs = tf.nn.softmax(rpn_class_logits, name='rpn_class_xxx')

    c1s1k4x = ops.conv2d(c3s1k512, 4 * anchors_per_location, 1, 0, 1, norm=None, activation=None,
                         reuse=reuse, is_training=is_training, name='rpn_bbox_pred',
                         use_bias=use_bias, kernel_initializer='glorot_uniform_tanh')

    rpn_bbox = tf.reshape(c1s1k4x, [tf.shape(c1s1k4x)[0], -1, 4])

    return [rpn_class_logits, rpn_probs, rpn_bbox]


def fpn_classifier(ipt, pool_size, num_classes, is_training=True,
                   fc_layers_size=1024):

    ipt = tf.map_fn(lambda x: ops.conv2d(x, fc_layers_size, pool_size, 0, 1, norm='batch', activation=tf.nn.relu,
                                         is_training=is_training, name='mrcnn_class_conv1',
                                         use_bias=True, kernel_initializer='glorot_uniform_tanh'),
                    elems=ipt, dtype=tf.float32)

    ipt = tf.map_fn(lambda x: ops.conv2d(x, fc_layers_size, 1, 0, 1, norm='batch', activation=tf.nn.relu,
                                         is_training=is_training, name='mrcnn_class_conv2',
                                         use_bias=True, kernel_initializer='glorot_uniform_tanh'),
                    elems=ipt, dtype=tf.float32)

    shared = tf.squeeze(tf.squeeze(ipt, 3), 2)

    mrcnn_class_logits = tf.map_fn(lambda x: ops.fully_connected(x, num_classes, name='mrcnn_class_logits',
                                                                 weights_initializer='glorot_uniform_tanh'),
                                   elems=shared, dtype=tf.float32)

    mrcnn_probs = tf.map_fn(lambda x: tf.nn.softmax(x, name='mrcnn_class'), elems=mrcnn_class_logits, dtype=tf.float32)

    ipt = tf.map_fn(lambda x: ops.fully_connected(x, 4 * num_classes, name='mrcnn_bbox_fc',
                                                  weights_initializer='glorot_uniform_tanh'),
                    elems=shared, dtype=tf.float32)

    ipt_shape = tf.shape(ipt)

    mrcnn_bbox = tf.reshape(ipt, [-1, ipt_shape[1], num_classes, 4])

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


def build_fpn_mask(ipt, num_classes, is_training=True):

    ipt = tf.map_fn(lambda x: ops.conv2d(x, 256, 3, 1, 1, norm='batch', activation=tf.nn.relu,
                                         is_training=is_training, name='mrcnn_mask_conv1',
                                         use_bias=True, kernel_initializer='glorot_uniform_tanh'),
                    elems=ipt, dtype=tf.float32)

    ipt = tf.map_fn(lambda x: ops.conv2d(x, 256, 3, 1, 1, norm='batch', activation=tf.nn.relu,
                                         is_training=is_training, name='mrcnn_mask_conv2',
                                         use_bias=True, kernel_initializer='glorot_uniform_tanh'),
                    elems=ipt, dtype=tf.float32)

    ipt = tf.map_fn(lambda x: ops.conv2d(x, 256, 3, 1, 1, norm='batch', activation=tf.nn.relu,
                                         is_training=is_training, name='mrcnn_mask_conv3',
                                         use_bias=True, kernel_initializer='glorot_uniform_tanh'),
                    elems=ipt, dtype=tf.float32)

    ipt = tf.map_fn(lambda x: ops.conv2d(x, 256, 3, 1, 1, norm='batch', activation=tf.nn.relu,
                                         is_training=is_training, name='mrcnn_mask_conv4',
                                         use_bias=True, kernel_initializer='glorot_uniform_tanh'),
                    elems=ipt, dtype=tf.float32)

    ipt = tf.map_fn(lambda x: ops.unconv2d(x, 256, 2, 2, norm=None, activation=tf.nn.relu,
                                           use_bias=True, name='mrcnn_mask_unconv', is_training=is_training,
                                           kernel_initializer='glorot_uniform_tanh'),
                    elems=ipt, dtype=tf.float32)

    opt = tf.map_fn(lambda x: ops.conv2d(x, num_classes, 1, 0, 1, activation=tf.nn.sigmoid,
                                         is_training=is_training, use_bias=True,
                                         kernel_initializer='glorot_uniform_tanh',
                                         name='mrcnn_mask'),
                    elems=ipt, dtype=tf.float32)

    return opt


def rpn_loss(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.

    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
                   -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG.
    """
    rpn_match = tf.squeeze(rpn_match, -1)

    anchor_class = tf.cast(tf.equal(rpn_match, 1), tf.int32)

    indices = tf.where(tf.not_equal(rpn_match, 0))

    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)

    anchor_class = tf.expand_dims(anchor_class, -1)
    anchor_class = tf.one_hot(anchor_class, 2, dtype=tf.float32, axis=-1)

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=anchor_class, logits=rpn_class_logits)

    loss = tf.cond(tf.size(loss) > 0, true_fn=lambda: tf.reduce_mean(loss),
                   false_fn=lambda: tf.constant(0.0))

    return loss


def rpn_bbox_loss(target_bbox, rpn_match, rpn_bbox):
    rpn_match = tf.squeeze(rpn_match, -1)
    indices = tf.where(tf.equal(rpn_match, 1))

    rpn_bbox = tf.gather_nd(rpn_bbox, indices)
    batch_counts = tf.reduce_sum(tf.cast(tf.equal(rpn_match, 1), tf.int32), axis=1)

    target_bbox = ops.batch_pack_graph(target_bbox, batch_counts,
                                       hyper_parameters.FLAGS.IMAGES_PER_GPU)

    loss = ops.smooth_l1_loss(target_bbox, rpn_bbox)

    loss = tf.cond(tf.size(loss) > 0, true_fn=lambda: tf.reduce_mean(loss),
                   false_fn=lambda: tf.constant(0.0))

    return loss


def mrcnn_class_loss(target_class_ids, pred_class_logits,
                     active_class_ids):
    target_class_ids = tf.cast(target_class_ids, 'int64')

    pred_class_ids = tf.argmax(pred_class_logits, axis=2)

    # pred_active = utils.batch_slice([active_class_ids, pred_class_ids], lambda x, y: tf.gather(x, y),
    #                                 hyper_parameters.FLAGS.IMAGE_PER_GPU)
    pred_active = tf.cast(tf.gather(active_class_ids[0], pred_class_ids), tf.float32)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_class_ids, logits=pred_class_logits)

    loss = loss * pred_active
    loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)
    return loss


def mrcnn_bbox_loss(target_bbox, target_class_ids, pred_bbox):
    target_class_ids = tf.reshape(target_class_ids, (-1,))
    target_bbox = tf.reshape(target_bbox, (-1, 4))
    pred_bbox = tf.reshape(pred_bbox, (-1, tf.shape(pred_bbox)[2], 4))

    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices)
    # loss = tf.keras.backend.switch(tf.size(target_bbox) > 0,
    #                                ops.smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
    #                                tf.constant(0.0))

    loss = ops.smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox)

    loss = tf.reduce_mean(loss)
    return loss


def mrcnn_mask_loss(target_masks, target_class_ids, pred_masks):
    target_class_ids = tf.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = tf.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    pred_shape = tf.shape(pred_masks)
    pred_masks = tf.reshape(pred_masks,
                            (-1, pred_shape[2], pred_shape[3], pred_shape[4]))

    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    target_masks = tf.gather(target_masks, positive_ix)
    pred_masks = tf.gather_nd(pred_masks, indices)

    # loss = tf.keras.backend.switch(tf.size(target_masks) > 0,
    #                                tf.keras.backend.binary_crossentropy(target=target_masks, output=pred_masks),
    #                                tf.constant(0.0))

    loss = target_masks * tf.log(pred_masks) + (1 - target_masks) * tf.log(pred_masks)

    loss = tf.reduce_mean(loss)
    return loss
