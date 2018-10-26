import layer
import utils
import tensorflow as tf


class ObjDetection:
    def __init__(self, image_per_gpu=1, gpu_count=1, detection_max_instances=100):
        self.image_per_gpu = image_per_gpu
        self.gpu_count = gpu_count
        self.detection_max_instances = detection_max_instances
        self.batch_size = self.image_per_gpu * self.gpu_count

    def __call__(self, ipt):
        rois = ipt[0]
        mrcnn_class = ipt[1]
        mrcnn_bbox = ipt[2]
        image_meta = ipt[3]

        m = utils.parse_image_meta_graph(image_meta)
        image_shape = m['image_shape'][0]
        window = utils.norm_boxes(m['window'], image_shape[:2])

        detections_batch = utils.batch_slice([rois, mrcnn_class, mrcnn_bbox, window],
                                             lambda w, x, y, z: layer.refine_detections(w, x, y, z),
                                             self.image_per_gpu)

        return tf.reshape(
            detections_batch,
            [self.image_per_gpu, self.detection_max_instances, 6])

    def compute_output_shape(self):
        return None, self.detection_max_instances, 6
