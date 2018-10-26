import layer
import utils


class TargetDetection:
    def __init__(self, mask_shape,
                 image_per_gpu=1,
                 train_rois_per_image=200,
                 ):
        self.image_per_gpu = image_per_gpu
        self.train_rois_per_image = train_rois_per_image
        self.mask_shape = mask_shape

    def __call__(self, ipt):
        proposals = ipt[0]
        gt_class_ids = ipt[1]
        gt_boxes = ipt[2]
        gt_masks = ipt[3]

        names = ["rois", "target_class_ids", "target_deltas", "target_mask"]
        outputs = utils.batch_slice(
            [proposals, gt_class_ids, gt_boxes, gt_masks],
            lambda w, x, y, z: layer.target_detection(
                w, x, y, z),
            self.image_per_gpu, names=names)
        return outputs

    def compute_output_shape(self):
        return [
            (None, self.train_rois_per_image, 4),  # rois
            (None, self.train_rois_per_image),  # class_ids
            (None, self.train_rois_per_image, 4),  # deltas
            (None, self.train_rois_per_image, self.mask_shape[0],
             self.mask_shape[1])  # masks
        ]
