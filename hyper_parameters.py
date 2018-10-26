import tensorflow as tf


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', 4, 'batch size, default: 1')
tf.flags.DEFINE_integer('image_size', 416, 'image size, default: 256')
tf.flags.DEFINE_float('learning_rate', 1e-3,
                      'initial learning rate for Adam, default: 0.001')
tf.flags.DEFINE_integer('num_id1', 1, 'number of dark_net_id1')
tf.flags.DEFINE_integer('num_id2', 2, 'number of dark_net_id2')
tf.flags.DEFINE_integer('num_id3', 8, 'number of dark_net_id3')
tf.flags.DEFINE_integer('num_id4', 8, 'number of dark_net_id4')
tf.flags.DEFINE_integer('num_id5', 4, 'number of dark_net_id5')
tf.flags.DEFINE_integer('max_num_boxes_per_image', 5, 'max number of boxes per image')
tf.flags.DEFINE_string('X', 'data/tfrecords/train.tfrecords',
                       'X tfrecords file for training, default: data/tfrecords/train.tfrecords')
tf.flags.DEFINE_string('labels_file', 'data/label/keypoint_train_annotations_20170909.json',
                       'labels file for training, default: data/label/keypoint_train_annotations_20170911.json')
tf.flags.DEFINE_integer('num_anchors', 9, 'the number of anchors, default: 9')
tf.flags.DEFINE_integer('num_classes', 1, 'the number of classes')
tf.flags.DEFINE_string('load_model', None,
                       'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')
tf.flags.DEFINE_float('threshold', 0.5, 'Impacts how the loss is calculated. '
                                        'Must be between zero and one, and the default is set to 0.5.')
tf.flags.DEFINE_string('norm', 'batch', 'norm method, default is batch')
tf.flags.DEFINE_string('anchors_path', 'data/anchors.txt', 'The path that points towards where '
                                                           'the anchor values for the model are stored.')


tf.flags.DEFINE_integer('PRE_NMS_LIMIT', 6000,
                        'ROIs kept after tf.nn.top_k and before non-maximum suppression, default is 6000.')
tf.flags.DEFINE_integer('IMAGES_PER_GPU', 1,
                        'Number of images to train with on each GPU. A 12GB GPU can typically '
                        'handle 2 images of 1024x1024px.')
tf.flags.DEFINE_integer('POST_NMS_ROIS_TRAINING', 2000,
                        'ROIs kept after non-maximum suppression (training).')
tf.flags.DEFINE_integer('POST_NMS_ROIS_INTERFACE', 1000,
                        'ROIs kept after non-maximum suppression (inference).')
tf.flags.DEFINE_float('RPN_NMS_THRESHOLD', 0.7,
                      'Non-max suppression threshold to filter RPN proposals.')
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
tf.flags.DEFINE_float('DETECTION_MIN_CONFIDENCE', 0.7,
                      'Minimum probability value to accept a detected instance.')
tf.flags.DEFINE_integer('DETECTION_MAX_INSTANCES', 100,
                        'Max number of final detections.')
tf.flags.DEFINE_float('DETECTION_NMS_THRESHOLD', 0.3,
                      'Non-maximum suppression threshold for detection.')
tf.flags.DEFINE_integer('GPU_COUNT', 1,
                        'NUMBER OF GPUs to use. When using only a CPU, '
                        'this needs to be set to 1.')
tf.flags.DEFINE_integer('ROI_SIZE', 7, 'Size of rois.')
tf.flags.DEFINE_integer('POOL_SIZE', 7, 'Size of pool.')
tf.flags.DEFINE_integer('MASK_POOL_SIZE', 14, 'Size of mask pool.')
