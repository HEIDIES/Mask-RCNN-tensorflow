import tensorflow as tf


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('model_dir', None,
                       'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')
tf.flags.DEFINE_string('pretrained_model_checkpoints', 'pretrained_model/resnet_v2_50.ckpt',
                       'pretrained resnet_v2_50 model file, default: pretrained_model/resnet_v2_50.ckpt.')

tf.flags.DEFINE_string('OPTIMIZER_MODE', 'Adam', 'Optimizer method.')
tf.flags.DEFINE_float('LEARNING_RATE', 0.001, 'Learning rate.')
tf.flags.DEFINE_float('LEARNING_MOMENTUM', 0.9, 'Learning momentum')
tf.flags.DEFINE_float('WEIGHT_DECAY', 0.0001, 'Weight decay regularization')


tf.flags.DEFINE_integer('PRE_NMS_LIMIT', 6000,
                        'ROIs kept after tf.nn.top_k and before non-maximum suppression, default is 6000.')
tf.flags.DEFINE_integer('POST_NMS_ROIS_TRAINING', 2000,
                        'ROIs kept after non-maximum suppression (training).')
tf.flags.DEFINE_integer('POST_NMS_ROIS_INFERENCE', 1000,
                        'ROIs kept after non-maximum suppression (inference).')


tf.flags.DEFINE_integer('IMAGES_PER_GPU', 1,
                        'Number of images to train with on each GPU. A 12GB GPU can typically '
                        'handle 2 images of 1024x1024px.')
tf.flags.DEFINE_integer('GPU_COUNT', 1,
                        'NUMBER OF GPUs to use. When using only a CPU, '
                        'this needs to be set to 1.')


tf.flags.DEFINE_float('RPN_NMS_THRESHOLD', 0.7,
                      'Non-max suppression threshold to filter RPN proposals.')
tf.flags.DEFINE_float('DETECTION_NMS_THRESHOLD', 0.3,
                      'Non-maximum suppression threshold for detection.')


tf.flags.DEFINE_integer('TRAIN_ROIS_PER_IMAGE', 200,
                        'Number of ROIs per image to feed to classifier/mask heads.')
tf.flags.DEFINE_float('ROI_POSITIVE_RATIO', 0.33,
                      'Percent of positive ROIs used to train classifier/mask heads.')


tf.flags.DEFINE_list('BBOX_STD_DEV', [0.1, 0.1, 0.2, 0.2],
                     'Bounding box refinement standard deviation for final detections.')
tf.flags.DEFINE_list('RPN_BBOX_STD_DEV', [0.1, 0.1, 0.2, 0.2],
                     'Bounding box refinement standard deviation for RPN.')


tf.flags.DEFINE_bool('USE_MINI_MASK', True,
                     ' If enabled, resizes instance masks to a smaller size to reduce '
                     'memory load.')
tf.flags.DEFINE_list('MASK_SHAPE', [28, 28], 'Shape of output mask.')
tf.flags.DEFINE_list('MINI_MASK_SHAPE', [56, 56], '[height, width] of the mini-mask')


tf.flags.DEFINE_float('DETECTION_MIN_CONFIDENCE', 0.7,
                      'Minimum probability value to accept a detected instance.')
tf.flags.DEFINE_integer('DETECTION_MAX_INSTANCES', 100,
                        'Max number of final detections.')
tf.flags.DEFINE_integer('MAX_GT_INSTANCES', 100, 'Maximum number of ground truth '
                                                 'instances to use in one image.')


tf.flags.DEFINE_integer('ROI_SIZE', 7, 'Size of rois.')
tf.flags.DEFINE_integer('POOL_SIZE', 7, 'Size of pool.')
tf.flags.DEFINE_integer('MASK_POOL_SIZE', 14, 'Size of mask pool.')


tf.flags.DEFINE_integer('IMAGE_MIN_DIM', 400, 'Minimal size for image resizing.')
tf.flags.DEFINE_integer('IMAGE_MAX_DIM', 512, 'Maximal size for image resizing.')
tf.flags.DEFINE_integer('TOP_DOWN_PYRAMID_SIZE', 256, 'Size of the top-down layers used '
                                                      'to build the feature pyramid')
tf.flags.DEFINE_integer('IMAGE_CHANNEL_COUNT', 3, 'Number of color channels per image. '
                                                  'RGB = 3, grayscale = 1, RGB-D = 4.')
tf.flags.DEFINE_float('IMAGE_MIN_SCALE', 0.0, 'Minimum scaling ratio. Checked after MIN_IMAGE_DIM '
                                              'and can force furtherup scaling.')
tf.flags.DEFINE_list('IMAGE_SHAPE',
                     [FLAGS.IMAGE_MAX_DIM, FLAGS.IMAGE_MAX_DIM, FLAGS.IMAGE_CHANNEL_COUNT],
                     'Input image size.')


tf.flags.DEFINE_string('IMAGE_RESIZE_MODE', 'square', 'Methods of image resizing.')


tf.flags.DEFINE_integer('NUM_CLASSES', 80 + 1, 'Number of classes.')


tf.flags.DEFINE_string('BACKBONE', 'resnet101', 'Backbone network architecture.')
tf.flags.DEFINE_list('BACKBONE_STRIDES', [4, 8, 16, 32, 64], 'The strides of each layer of the FPN Pyramid.')
tf.flags.DEFINE_bool('USE_PRETRAINED_MODEL', True, 'Use pretrained resnet-50 weights.')
tf.flags.DEFINE_bool('STAGE5', True, 'Use stage5.')


tf.flags.DEFINE_integer('RPN_ANCHOR_STRIDE', 1, 'Anchor stride.')
tf.flags.DEFINE_list('RPN_ANCHOR_SCALES', [32, 64, 128, 256, 512], 'Length of square anchor side in pixels')
tf.flags.DEFINE_list('RPN_ANCHOR_RATIOS', [0.5, 1, 2], 'Ratios of anchors at each cell (width/height).')
tf.flags.DEFINE_integer('RPN_TRAIN_ANCHORS_PER_IMAGE', 256, 'How many anchors per image '
                                                            'to use for RPN training.')
tf.flags.DEFINE_bool('USE_RPN_ROIS', True, 'Use RPN ROIs or externally generated ROIs for training.')


tf.flags.DEFINE_string('norm', 'batch', 'Norm method, default is batch')
tf.flags.DEFINE_bool('use_bias', True, 'Use bias or not.')
tf.flags.DEFINE_string('mode', 'training', 'trianing or interface.')

tf.flags.DEFINE_list('MEAN_PIXEL', [123.7, 116.8, 103.9], 'Image mean (RGB).')
