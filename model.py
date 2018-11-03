import tensorflow as tf
from resnet import Resnet
from rpn import RPN
from obj_detection import ObjDetection
from target_detection import TargetDetection
from fpn_classifier import FpnClassifier
from fpn_mask import FpnMask
from proposal import ProposalLayer
from fpn import FPN
# from dataset import  DataSet
from datetime import datetime
from hyper_parameters import FLAGS
import layer
from data_generator import data_generator
import utils
from pyramid_roi_pooling import PyramidRoiPooling
from tensorflow.contrib.slim import nets
import numpy as np
import os
import logging

slim = tf.contrib.slim


class MRCNN:
    def __init__(self, mode, rpn_anchor_ratios, rpn_anchor_scales, mask_shape, pool_size,
                 image_shape, mini_mask_shape, backbone_strides, mean_pixel,
                 roi_size=7, backbone='resnet50', stage5=True,
                 norm='batch', use_bias=True, rpn_anchor_stride=1, image_per_gpu=1,
                 gpu_count=1, detection_max_instances=100, train_rois_per_image=200,
                 num_classes=1, use_mini_mask=True, use_pretrained_model=True,
                 top_down_pyramid_size=256, post_nms_rois_training=2000,
                 post_nms_rois_inference=1000, pre_nms_limit=6000, rpn_nms_threshold=0.7,
                 use_rpn_rois=True, model_dir=None,
                 optimizer_method='Adam', learning_rate=0.001, momentum=0.9,
                 weight_decay=0.0001, image_min_dim=800, image_max_dim=1024,
                 image_min_scale=0.0, image_resize_mode='square',
                 max_gt_instances=100, rpn_train_anchors_per_image=256):

        assert mode in ['training', 'inference']
        assert optimizer_method in ['Adam', 'SGD']

        tf.reset_default_graph()
        self.graph = tf.Graph()

        self.mode = mode
        self.rpn_anchor_ratios = rpn_anchor_ratios
        self.rpn_anchor_scales = rpn_anchor_scales
        self.mask_shape = mask_shape
        self.pool_size = pool_size
        self.image_shape = np.array(image_shape)
        self.mini_mask_shape = mini_mask_shape
        self.backbone_strides = backbone_strides
        self.mean_pixel = mean_pixel

        self.roi_size = roi_size
        self.backbone = backbone
        self.stage5 = stage5
        self.norm = norm
        self.use_bias = use_bias
        self.rpn_anchor_stride = rpn_anchor_stride
        self.image_per_gpu = image_per_gpu
        self.gpu_count = gpu_count
        self.detection_max_instances = detection_max_instances
        self.train_rois_per_image = train_rois_per_image
        self.num_classes = num_classes
        self.use_mini_mask = use_mini_mask
        self.use_pretrained_model = use_pretrained_model
        self.top_down_pyramid_size = top_down_pyramid_size
        self.post_nms_rois_training = post_nms_rois_training
        self.post_nms_rois_inference = post_nms_rois_inference
        self.pre_nms_limit = pre_nms_limit
        self.rpn_nms_threshold = rpn_nms_threshold
        self.use_rpn_rois = use_rpn_rois
        self.model_dir = model_dir
        self.optimizer_method = optimizer_method
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.image_min_dim = image_min_dim
        self.image_max_dim = image_max_dim
        self.image_min_scale = image_min_scale
        self.image_resize_mode = image_resize_mode
        self.max_gt_instances = max_gt_instances
        self.rpn_train_anchors_per_image = rpn_train_anchors_per_image

        self.image_meta_size = 1 + 3 + 3 + 4 + 1 + self.num_classes
        self.reuse = False
        self._anchor_cache = {}
        self.batch_size = self.gpu_count * self.image_per_gpu
        self.backbone_shape = utils.compute_backbone_shapes(self.backbone, self.backbone_strides, self.image_shape)
        self.num_anchors_per_image = len(self.rpn_anchor_ratios) * (self.backbone_shape[0][0] * self.backbone_shape[0][0] +
                                                                    self.backbone_shape[1][0] * self.backbone_shape[1][0] +
                                                                    self.backbone_shape[2][0] * self.backbone_shape[2][0] +
                                                                    self.backbone_shape[3][0] * self.backbone_shape[3][0] +
                                                                    self.backbone_shape[4][0] * self.backbone_shape[4][0])

        with self.graph.as_default():

            self.is_training = tf.placeholder_with_default(False, [])
            self.input_image = tf.placeholder(dtype=tf.float32,
                                              shape=[None, self.image_shape[0], self.image_shape[1],
                                                     self.image_shape[2]],
                                              name='input_image')
            self.input_image_meta = tf.placeholder(dtype=tf.int32,
                                                   shape=[None, self.image_meta_size],
                                                   name='input_image_meta')

            if mode == 'training':
                self.input_rpn_match = tf.placeholder(dtype=tf.int32,
                                                      shape=[None, self.num_anchors_per_image, 1],
                                                      name='input_rpn_match')
                self.input_rpn_boxes = tf.placeholder(dtype=tf.float32,
                                                      shape=[None, self.rpn_train_anchors_per_image, 4],
                                                      name='input_rpn_boxes')
                self.input_gt_class_ids = tf.placeholder(dtype=tf.int32,
                                                         shape=[None, self.max_gt_instances], name='input_gt_class_ids')
                self.input_gt_boxes = tf.placeholder(dtype=tf.float32,
                                                     shape=[None, self.max_gt_instances, 4], name='input_gt_boxes')
                self.input_gt_boxes_normalized = utils.norm_boxes_graph(self.input_gt_boxes,
                                                                        tf.shape(self.input_image)[1:3])
                self.proposal_count = self.post_nms_rois_training
                if self.use_mini_mask:
                    self.input_gt_masks = tf.placeholder(dtype=tf.bool,
                                                         shape=[None, self.mini_mask_shape[0],
                                                                self.mini_mask_shape[1], self.max_gt_instances],
                                                         name='input_gt_mask')
                else:
                    self.input_gt_masks = tf.placeholder(dtype=tf.bool,
                                                         shape=[None, self.image_shape[0], self.image_shape[1],
                                                                self.max_gt_instances],
                                                         name='input_gt_mask')

            elif mode == 'inference':
                self.input_anchors = tf.placeholder(dtype=tf.float32, shape=[None, None, 4], name='input_anchors')
                self.proposal_count = self.post_nms_rois_inference

            self.resnet = Resnet(name='resnet', architecture=self.backbone, is_training=self.is_training,
                                 stage5=self.stage5, use_bias=self.use_bias)

            arg_scope = nets.resnet_v2.resnet_arg_scope()
            with slim.arg_scope(arg_scope):
                _, self.end_points = nets.resnet_v2.resnet_v2_50(self.input_image, num_classes=None,
                                                                 is_training=self.is_training)

            self.fpn = FPN(name='fpn', top_down_pyramid_size=self.top_down_pyramid_size, use_bias=self.use_bias)

            self.rpn = RPN(name='rpn', anchors_per_location=len(self.rpn_anchor_ratios),
                           anchor_stride=self.rpn_anchor_stride, is_training=self.is_training, use_bias=self.use_bias)
            self.proposal = ProposalLayer(self.pre_nms_limit, self.proposal_count, self.rpn_nms_threshold,
                                          self.image_per_gpu)
            self.pyramidRoiPooling = PyramidRoiPooling(name='PyramidRoiPooling', roi_size=self.roi_size)
            self.objDetection = ObjDetection(image_per_gpu=self.image_per_gpu, gpu_count=self.gpu_count,
                                             detection_max_instances=self.detection_max_instances)
            self.targetDetection = TargetDetection(mask_shape=self.mask_shape, image_per_gpu=self.image_per_gpu,
                                                   train_rois_per_image=self.train_rois_per_image)
            self.fpnClassifier = FpnClassifier('FpnClassifier', pool_size=self.pool_size, num_classes=self.num_classes,
                                               is_training=self.is_training)
            self.fpnMask = FpnMask('FpnMask', num_classes=self.num_classes, is_training=self.is_training)

    def get_anchors(self, image_shape):
        backbone_shapes = utils.compute_backbone_shapes(self.backbone, self.backbone_strides, image_shape)
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            a = utils.generate_pyramid_anchors(
                self.rpn_anchor_scales,
                self.rpn_anchor_ratios,
                backbone_shapes,
                self.backbone_strides,
                self.rpn_anchor_stride
            )
            self._anchor_cache[tuple(image_shape)] = utils.norm_boxes(a, image_shape[:2])
        return self._anchor_cache[tuple(image_shape)]

    def model(self):

        h, w = self.image_shape[:2]
        if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")
        if self.use_pretrained_model:

            c2, c3, c4, c5 = \
                self.end_points['resnet_v2_50/block1/unit_2/bottleneck_v2'], \
                self.end_points['resnet_v2_50/block2/unit_3/bottleneck_v2'], \
                self.end_points['resnet_v2_50/block3/unit_4/bottleneck_v2'], \
                self.end_points['resnet_v2_50/block4']

        else:

            if callable(self.backbone):
                _, c2, c3, c4, c5 = self.backbone(self.input_image,
                                                  stage5=self.stage5, is_training=self.is_training)

            else:
                _, c2, c3, c4, c5 = self.resnet(self.input_image)

        p2, p3, p4, p5, p6 = self.fpn([c2, c3, c4, c5])

        rpn_feature_maps = [p2, p3, p4, p5, p6]
        mrcnn_feature_maps = [p2, p3, p4, p5]

        if self.mode == 'training':
            anchors = self.get_anchors(self.image_shape)
            anchors = np.broadcast_to(anchors, (self.batch_size,) + anchors.shape)
            anchors = tf.constant(anchors)
        else:
            anchors = self.input_anchors

        layer_outputs = []
        for p in rpn_feature_maps:
            layer_outputs.append(self.rpn(p))

        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [tf.concat(list(o), name=n, axis=1) for o, n in zip(outputs, output_names)]

        rpn_class_logits, rpn_class, rpn_bbox = outputs
        rpn_rois = self.proposal([rpn_class, rpn_bbox, anchors])

        if self.mode == 'training':
            active_class_ids = utils.parse_image_meta_graph(self.input_image_meta)['active_class_ids']

            if not self.use_rpn_rois:
                input_rois = tf.placeholder(dtype=tf.int32, shape=[None, self.post_nms_rois_training, 4],
                                            name='input_rois')
                target_rois = utils.norm_boxes_graph(input_rois, tf.shape(self.input_image)[1:3])

            else:

                target_rois = rpn_rois

            rois, target_class_ids, target_bbox, target_mask = \
                self.targetDetection([target_rois, self.input_gt_class_ids,
                                      self.input_gt_boxes_normalized, self.input_gt_masks])

            pooled = self.pyramidRoiPooling([rois, self.input_image_meta] + mrcnn_feature_maps)
            pooled_mask = self.pyramidRoiPooling([rois, self.input_image_meta] + mrcnn_feature_maps, pool_size=14)

            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.fpnClassifier(pooled)
            mrcnn_mask = self.fpnMask(pooled_mask)

            output_rois = tf.identity(rois, name='output_rois')

            rpn_class_loss = layer.rpn_loss(self.input_rpn_match, rpn_class_logits)
            rpn_bbox_loss = layer.rpn_bbox_loss(self.input_rpn_boxes, self.input_rpn_match, rpn_bbox)
            class_loss = layer.mrcnn_class_loss(target_class_ids, mrcnn_class_logits, active_class_ids)
            bbox_loss = layer.mrcnn_bbox_loss(target_bbox, target_class_ids, mrcnn_bbox)
            mask_loss = layer.mrcnn_mask_loss(target_mask, target_class_ids, mrcnn_mask)

            tf.summary.scalar('rpn_class_loss', rpn_class_loss)
            tf.summary.scalar('rpn_bbox_loss', rpn_bbox_loss)
            tf.summary.scalar('mrcnn_class_loss', class_loss)
            tf.summary.scalar('mrcnn_bbox_loss', bbox_loss)
            tf.summary.scalar('mrcnn_mask_loss', mask_loss)

            outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                       mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
                       rpn_rois, output_rois,
                       rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]
        else:
            pooled = self.pyramidRoiPooling([rpn_rois, self.input_image_meta] + mrcnn_feature_maps)
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.fpnClassifier(pooled)

            detections = self.objDetection([rpn_rois, mrcnn_class, mrcnn_bbox, self.input_image_meta])

            detections_bbox = detections[..., :4]

            pooled = self.pyramidRoiPooling([detections_bbox, self.input_image_meta] + mrcnn_feature_maps, pool_size=14)

            mrcnn_mask = self.fpnMask(pooled)

            outputs = [detections, mrcnn_class, mrcnn_bbox, mrcnn_mask, rpn_rois, rpn_class, rpn_bbox]

        return outputs

    def mrcnn_optimizer(self, total_loss, var_lists, optimizer_method='Adam'):
        def make_optimizer(loss, variables):
            tf.summary.scalar('learning_rate/{}'.format(optimizer_method), self.learning_rate)
            learning_step = (
                tf.train.AdamOptimizer(self.learning_rate, name=optimizer_method).
                minimize(loss, var_list=variables)
            )

            return learning_step

        fpn_var_list = self.fpn.var_list
        rpn_var_list = self.rpn.var_list
        fpn_classifier_var_list = self.fpnClassifier.var_list
        fpn_mask_var_list = self.fpnMask.vat_list

        if self.use_pretrained_model:
            resnet_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='resnet_v2_50')
        else:
            resnet_var_list = self.resnet.var_list

        var_list = resnet_var_list + fpn_var_list + rpn_var_list + fpn_classifier_var_list + fpn_mask_var_list

        reg_losses = [self.weight_decay * tf.reduce_sum(tf.square(w)) / tf.cast(tf.size(w), tf.float32)
                      for w in var_list if 'gamma' not in w.name and 'beta' not in w.name]

        total_loss += tf.add_n(reg_losses)

        mrcnn_optimizer = make_optimizer(total_loss, var_lists)
        with tf.control_dependencies([mrcnn_optimizer]):
            return tf.no_op(name='optimizer')

    def set_trainable_var_list(self):

        if FLAGS.USE_PRETRAINED_MODEL:
            resnet_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='resnet_v2_50')
        else:
            resnet_var_list = self.resnet.var_list
        heads_var_list = self.fpn.var_list + self.fpnClassifier.var_list + self.fpnMask.vat_list + self.rpn.var_list
        all_var_list = heads_var_list + resnet_var_list
        res3_up_var_list = [w for w in all_var_list if 'res2' not in w.name or 'resnet/conv1' not in w.name]
        res4_up_var_list = [w for w in res3_up_var_list if 'res3' not in w.name]
        res5_up_var_list = [w for w in res4_up_var_list if 'res4' not in w.name]

        if FLAGS.USE_PRETRAINED_MODEL:
            res3_up_var_list = heads_var_list + [w for w in resnet_var_list if 'resnet_v2_50/block2' in w.name
                                                 or 'resnet_v2_50/block3' in w.name
                                                 or 'resnet_v2_50/block4' in w.name]
            res4_up_var_list = heads_var_list + [w for w in resnet_var_list if 'resnet_v2_50/block3' in w.name
                                                 or 'resnet_v2_50/block4' in w.name]
            res5_up_var_list = heads_var_list + [w for w in resnet_var_list if 'resnet_v2_50/block4' in w.name]

        return {
            'all': all_var_list,
            'heads': heads_var_list,
            '3+': res3_up_var_list,
            '4+': res4_up_var_list,
            '5+': res5_up_var_list
        }

    def train(self, train_dataset, val_dataset, epochs, layers_var_list, learning_rate=None,
              augmentation=None, no_augmentation_sources=None, mode='training'):
        assert mode == 'training'

        if learning_rate is not None:
            self.learning_rate = learning_rate

        if FLAGS.model_dir is not None:
            checkpoints_dir = "checkpoints/" + FLAGS.model_dir.lstrip("checkpoints/")
        else:
            current_time = datetime.now().strftime("%Y%m%d-%H%M")
            checkpoints_dir = "checkpoints/{}".format(current_time)
            try:
                os.makedirs(checkpoints_dir)
            except os.error:
                pass

        with self.graph.as_default():

            train_generator = data_generator(train_dataset, shuffle=True,
                                             augmentation=augmentation,
                                             batch_size=self.batch_size,
                                             no_augmentation_sources=no_augmentation_sources)

            val_generator = data_generator(val_dataset, shuffle=True,
                                           batch_size=self.batch_size)

            outputs = self.model()

            self.trainable_var_list = self.set_trainable_var_list()

            rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss = outputs[9:]

            total_loss = rpn_class_loss + rpn_bbox_loss + class_loss + bbox_loss + mask_loss
            optimizer = self.mrcnn_optimizer(total_loss=total_loss, var_lists=self.trainable_var_list[layers_var_list])

            resnet_saver = []

            if FLAGS.USE_PRETRAINED_MODEL:
                resnet_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='resnet_v2_50')
                resnet_saver = tf.train.Saver(var_list=resnet_var_list)

            saver = tf.train.Saver()

        config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        with self.graph.as_default():
            summary_op = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(checkpoints_dir, self.graph)

            with tf.Session(graph=self.graph, config=config) as sess:
                if FLAGS.model_dir is not None:
                    checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
                    meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
                    restore = tf.train.import_meta_graph(meta_graph_path)
                    restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
                    step = int(meta_graph_path.split("-")[2].split(".")[0])
                else:
                    sess.run(tf.global_variables_initializer())
                    if FLAGS.USE_PRETRAINED_MODEL:
                        resnet_saver.restore(sess, FLAGS.pretrained_model_checkpoints)
                    step = 0

                num_iters = len(train_dataset.image_ids) * epochs // self.batch_size

                while step < num_iters:
                    inputs_train = train_generator.__next__()
                    _, total_loss_val, summary = sess.run([optimizer, total_loss, summary_op],
                                                          feed_dict={self.input_image: inputs_train[0][0],
                                                                     self.input_image_meta: inputs_train[0][1],
                                                                     self.input_rpn_match: inputs_train[0][2],
                                                                     self.input_rpn_boxes: inputs_train[0][3],
                                                                     self.input_gt_class_ids: inputs_train[0][4],
                                                                     self.input_gt_boxes: inputs_train[0][5],
                                                                     self.input_gt_masks: inputs_train[0][6]})

                    if (step + 1) % 100 == 0:
                        logging.info('-----------Step %d:-------------' % (step + 1))
                        logging.info('  total_train_loss_val   : {}'.format(total_loss_val))

                    if (step + 1) % 10000 == 0:
                        inputs_validation = val_generator.__next__()
                        total_loss_val = sess.run([total_loss],
                                                  feed_dict={self.input_image: inputs_validation[0][0],
                                                             self.input_image_meta: inputs_validation[0][1],
                                                             self.input_rpn_match: inputs_validation[0][2],
                                                             self.input_rpn_boxes: inputs_validation[0][3],
                                                             self.input_gt_class_ids: inputs_validation[0][4],
                                                             self.input_gt_boxes: inputs_validation[0][5],
                                                             self.input_gt_masks: inputs_validation[0][6]})

                        train_writer.add_summary(summary, step)

                        save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
                        logging.info("Model saved in file: %s" % save_path)
                        logging.info('-----------Step %d:-------------' % (step + 1))
                        logging.info('  total_val_loss_val   : {}'.format(total_loss_val))

                    step += 1

                save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
                logging.info("Model saved in file: %s" % save_path)

    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matrices [height,width,depth]. Images can have
            different sizes.

        Returns 3 Numpy matrices:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # Resize image
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=self.image_min_dim,
                min_scale=self.image_min_scale,
                max_dim=self.image_max_dim,
                mode=self.image_resize_mode)
            molded_image = utils.mold_image(molded_image, np.array(self.mean_pixel))
            # Build image_meta
            image_meta = utils.compose_image_meta(
                0, image.shape, molded_image.shape, window, scale,
                np.zeros([self.num_classes], dtype=np.int32))
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def unmold_detections(self, detections, mrcnn_mask, original_image_shape,
                          image_shape, window):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        detections: [N, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
        mrcnn_mask: [N, height, width, num_classes]
        original_image_shape: [H, W, C] Original image shape before resizing
        image_shape: [H, W, C] Shape of the image after resizing and padding
        window: [y1, x1, y2, x2] Pixel coordinates of box in the image where the real
                image is excluding the padding.

        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]

        # Translate normalized coordinates in the resized image to pixel
        # coordinates in the original image before resizing
        window = utils.norm_boxes(window, image_shape[:2])
        wy1, wx1, wy2, wx2 = window
        shift = np.array([wy1, wx1, wy1, wx1])
        wh = wy2 - wy1  # window height
        ww = wx2 - wx1  # window width
        scale = np.array([wh, ww, wh, ww])
        # Convert boxes to normalized coordinates on the window
        boxes = np.divide(boxes - shift, scale)
        # Convert boxes to pixel coordinates on the original image
        boxes = utils.denorm_boxes(boxes, original_image_shape[:2])

        # Filter out detections with zero area. Happens in early training when
        # network weights are still random
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = utils.unmold_mask(masks[i], boxes[i], original_image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1)\
            if full_masks else np.empty(original_image_shape[:2] + (0,))

        return boxes, class_ids, scores, full_masks

    def detect(self, images, verbose=0):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert FLAGS.model_dir is not None
        assert len(
            images) == self.batch_size, "len(images) must be equal to BATCH_SIZE"

        if verbose:
            utils.log("Processing {} images".format(len(images)))
            for image in images:
                utils.log("image", image)

        molded_images, image_metas, windows = self.mold_inputs(images)

        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape, \
                "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

        anchors = self.get_anchors(image_shape)

        anchors = np.broadcast_to(anchors, (self.batch_size,) + anchors.shape)

        if verbose:
            utils.log("molded_images", molded_images)
            utils.log("image_metas", image_metas)
            utils.log("anchors", anchors)

        detections_, _, _, mrcnn_mask_, _, _, _ = \
            self.model()

        checkpoints_dir = "checkpoints/" + FLAGS.model_dir.lstrip("checkpoints/")

        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as sess:
                checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
                meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
                restore = tf.train.import_meta_graph(meta_graph_path)
                restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))

                detections, mrcnn_mask = sess.run([detections_, mrcnn_mask_], feed_dict={
                    self.input_image: molded_images,
                    self.input_image_meta: image_metas,
                    self.input_anchors: anchors
                })

                results = []
                for i, image in enumerate(images):
                    final_rois, final_class_ids, final_scores, final_masks = \
                        self.unmold_detections(detections[i], mrcnn_mask[i],
                                               image.shape, molded_images[i].shape,
                                               windows[i])
                    results.append({
                        "rois": final_rois,
                        "class_ids": final_class_ids,
                        "scores": final_scores,
                        "masks": final_masks,
                    })
        return results
