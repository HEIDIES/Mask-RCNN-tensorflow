import tensorflow as tf
import layer
import utils
import numpy as np


class ProposalLayer:
    def __init__(self, pre_nms_limit=6000,
                 post_nms_rois_limit=2000,
                 rpn_nms_threshold=0.7,
                 image_per_gpu=1):

        self.pre_nms_limit = pre_nms_limit
        self.post_nms_rois_limit = post_nms_rois_limit
        self.rpn_nms_threshold = rpn_nms_threshold
        self.image_per_gpu = image_per_gpu

    def __call__(self, ipt):

        scores = ipt[0][:, :, 1]
        deltas = ipt[1]
        anchors = ipt[2]

        pre_nms_limit = tf.minimum(self.pre_nms_limit, tf.shape(anchors)[1])
        top_k_indices = tf.nn.top_k(scores, pre_nms_limit, name="top_anchors").indices

        top_k_scores = utils.batch_slice([scores, top_k_indices], lambda x, y: tf.gather(x, y),
                                         self.image_per_gpu)

        top_k_anchors = utils.batch_slice([anchors, top_k_indices], lambda x, y: tf.gather(x, y),
                                          self.image_per_gpu,
                                          names=["pre_nms_anchors"])

        top_k_deltas = utils.batch_slice([deltas, top_k_indices], lambda x, y: tf.gather(x, y),
                                         self.image_per_gpu)

        top_k_boxes = utils.batch_slice([top_k_anchors, top_k_deltas],
                                        lambda x, y: layer.apply_box_deltas_graph(x, y),
                                        self.image_per_gpu,
                                        names=["refined_anchors"])

        window = np.array([0, 0, 1, 1], dtype=np.float32)
        top_k_boxes = utils.batch_slice(top_k_boxes,
                                        lambda x: layer.clip_boxes_graph(x, window),
                                        self.image_per_gpu,
                                        names=["refined_anchors_clipped"])

        def nms(boxes, scores_):
            indices = tf.image.non_max_suppression(
                boxes, scores_, self.post_nms_rois_limit,
                self.rpn_nms_threshold, name="rpn_non_max_suppression")

            proposals = tf.gather(boxes, indices)
            padding = tf.maximum(self.post_nms_rois_limit - tf.shape(proposals)[0], 0)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            return proposals

        proposal_rois = utils.batch_slice([top_k_boxes, top_k_scores], nms,
                                          self.image_per_gpu)

        return proposal_rois

    def compute_output_shape(self):
        return None, self.post_nms_rois_limit, 4
