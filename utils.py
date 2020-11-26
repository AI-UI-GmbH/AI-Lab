import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as KL


def crop_and_resize(images, boxes, box_inds, crop_size):
    boxes = tf.stop_gradient(boxes)

    def transform_fpcoor_for_tf(boxes, image_shape, crop_shape):
        x0, y0, x1, y1 = tf.split(boxes, 4, axis=1)

        spacing_w = (x1 - x0) / tf.cast(crop_shape[1], dtype=tf.float32)
        spacing_h = (y1 - y0) / tf.cast(crop_shape[0], dtype=tf.float32)

        nx0 = (x0 + spacing_w / 2 - 0.5) / tf.cast(image_shape[1] - 1, dtype=tf.float32)
        ny0 = (y0 + spacing_h / 2 - 0.5) / tf.cast(image_shape[0] - 1, dtype=tf.float32)

        nw = spacing_w * tf.cast(crop_shape[1] - 1, dtype=tf.float32) / tf.cast(image_shape[1] - 1,
                                                                                dtype=tf.float32)
        nh = spacing_h * tf.cast(crop_shape[0] - 1, dtype=tf.float32) / tf.cast(image_shape[0] - 1,
                                                                                dtype=tf.float32)

        return tf.concat([ny0, nx0, ny0 + nh, nx0 + nw], axis=1)

    image_shape = tf.shape(images)[1:]
    boxes = transform_fpcoor_for_tf(boxes, image_shape, crop_size)
    ret = tf.image.crop_and_resize(images, boxes, box_inds, crop_size=crop_size)
    return ret


def roi_align(images, boxes, pool_shape):
    # boxes = tf.concat(boxes, axis=1)
    # TODO
    # box_inds = np.arange(self.cfg.BATCH_SIZE)
    # box_inds = np.repeat(box_inds, self.cfg.MAX_GT_INSTANCES)
    # box_inds = tf.convert_to_tensor(box_inds, dtype=tf.int32)
    num_images = images.shape[0]
    batch_size = boxes.shape[0]
    boxes = tf.reshape(boxes, [-1, 4])
    num_box = boxes.shape[0]

    if num_images != batch_size:
        num_box = num_box // num_images
        batch_size = num_images
    else:
        num_box = num_box // batch_size
    
    box_inds = tf.repeat(tf.range(0, batch_size, delta=1, dtype=tf.int32, name='range'), num_box)
    
    ret = crop_and_resize(images, boxes, box_inds, pool_shape)
    return ret


class ROIAlign(KL.Layer):
    def __init__(self, cfg, **kwargs):
        super(ROIAlign, self).__init__(**kwargs)
        self.cfg = cfg
        self.pool_shape = (cfg.ROI_RESOLUTION, cfg.ROI_RESOLUTION)

    def crop_and_resize(self, image, boxes, box_inds, crop_size):
        """
        Aligned version of tf.image.crop_and_resize, following our definition of floating point boxes.
        Args:
            image: NCHW
            boxes: nx4, x1y1x2y2
            box_inds: (n,)
            crop_size (tuple):
        Returns:
            n,C,size,size
        """
        boxes = tf.stop_gradient(boxes)

        def transform_fpcoor_for_tf(boxes, image_shape, crop_shape):
            x0, y0, x1, y1 = tf.split(boxes, 4, axis=1)

            spacing_w = (x1 - x0) / tf.cast(crop_shape[1], dtype=tf.float32)
            spacing_h = (y1 - y0) / tf.cast(crop_shape[0], dtype=tf.float32)

            nx0 = (x0 + spacing_w / 2 - 0.5) / tf.cast(image_shape[1] - 1, dtype=tf.float32)
            ny0 = (y0 + spacing_h / 2 - 0.5) / tf.cast(image_shape[0] - 1, dtype=tf.float32)

            nw = spacing_w * tf.cast(crop_shape[1] - 1, dtype=tf.float32) / tf.cast(image_shape[1] - 1,
                                                                                    dtype=tf.float32)
            nh = spacing_h * tf.cast(crop_shape[0] - 1, dtype=tf.float32) / tf.cast(image_shape[0] - 1,
                                                                                    dtype=tf.float32)

            return tf.concat([ny0, nx0, ny0 + nh, nx0 + nw], axis=1)

        image_shape = tf.shape(image)[1:]
        boxes = transform_fpcoor_for_tf(boxes, image_shape, crop_size)
        ret = tf.image.crop_and_resize(image, boxes, box_inds, crop_size=crop_size)
        return ret

    def call(self, inputs):
        images = inputs[0]
        boxes = inputs[1]
        boxes = tf.concat(boxes, axis=1)
        boxes = tf.reshape(boxes, [-1, 4])

        # box_inds = np.arange(self.cfg.BATCH_SIZE)
        # box_inds = np.repeat(box_inds, self.cfg.MAX_GT_INSTANCES)
        # box_inds = tf.convert_to_tensor(box_inds, dtype=tf.int32)
        box_inds = tf.zeros([tf.shape(boxes)[0]], dtype=tf.int32)

        ret = self.crop_and_resize(images, boxes, box_inds, self.pool_shape)
        # ret = tf.nn.avg_pool(ret, [1, 1, 2, 2], [1, 1, 2, 2], padding='SAME', data_format='NHWC')
        return ret

    def compute_output_shape(self, input_shape):
        return (input_shape[1][0] * input_shape[1][1]) + self.pool_shape + (4)


def conv_block(input_tensor, filters=128, kernel_size=3, strides=1, padding="same", use_bias=True, **kwargs):
    x = KL.Conv2D(filters=filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding=padding,
                  use_bias=use_bias,
                  **kwargs)(input_tensor)
    x = KL.BatchNormalization()(x)  # TODO: use group norm
    x = KL.Activation('relu')(x)
    return x


# This function is from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/utils.py
def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [num_instances, height, width]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[0], 4], dtype=np.float32)
    for i in range(mask.shape[0]):
        m = mask[i, :, :]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.float32)


def mold_image(images, config):
    """Expects an RGB image (or array of images) and subtracts
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.DATA.MEAN_PIXEL


def get_backbone(backbone, **kwargs):
    """return backbone as keras model

    Arguments:
        backbone: str or keras model
    """
    if isinstance(backbone, str):
        assert backbone in ['ResNet50V2', 'ResNet101V2', 'ResNet152V2', 'ResNet50', 'ResNet101', 'ResNet152']
        backbone = getattr(tf.keras.applications, backbone)(**kwargs)

    return backbone


