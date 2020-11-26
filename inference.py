import numpy as np
import tensorflow as tf

from config import get_config
from models.blendmask import BlendMask


def ltrb2boxes(centers, ltbr):
    boxes = tf.concat([
        centers - ltbr[:, :2],
        centers + ltbr[:, 2:]], axis=-1)
    return boxes


def decode_predictions(logits, score_threshold=0.6, centers=None):
    cls_target = tf.concat(logits[0], axis=1)
    ctr_target = tf.concat(logits[1], axis=1)
    reg_target = tf.concat(logits[2], axis=1)
    print(cls_target.shape, ctr_target.shape, reg_target.shape)
    cls_target = tf.sigmoid(cls_target)
    ctr_target = tf.sigmoid(ctr_target)
    print(cls_target.shape, ctr_target.shape)
    cls_scores = tf.reduce_max(cls_target[0], axis=1)
    print(cls_scores.shape)
    cls_ids = tf.argmax(cls_target[0], axis=1)
    print(cls_ids.shape)
    score_map = cls_scores * ctr_target[0, :, 0]
    print(score_map.shape)
    valid_indices = tf.where(score_map > score_threshold)[:, 0]
    print(valid_indices.shape)
    valid_scores = tf.gather(score_map, valid_indices)
    print(valid_scores.shape)
    valid_cls_ids = tf.gather(cls_ids, valid_indices)
    print(valid_cls_ids.shape)

    valid_centers = tf.gather(centers, valid_indices)
    valid_ltrb = tf.gather(reg_target[0], valid_indices)

    decoded_boxes = ltrb2boxes(valid_centers, valid_ltrb)

    nms_indices = tf.image.non_max_suppression(decoded_boxes,
                                               valid_scores,
                                               max_output_size=300)
    boxes = tf.gather(decoded_boxes, nms_indices)
    scores = tf.gather(valid_scores, nms_indices)
    ids = tf.gather(valid_cls_ids, nms_indices)
    return boxes, scores, ids


if __name__ == "__main__":
    
    cfg = get_config('blendmask')

    assert cfg.IMAGE_SIZE % 2 ** 6 == 0, "Image size must be dividable by 2 at least 6 times " \
                                         "to avoid fractions when downscaling and upscaling. " \
                                         "For example, use 256, 320, 384, 448, 512, ... etc."

    # build model
    blendmask = BlendMask(cfg)
    print("-----------------------------------")
    print("----------model built--------------")
    print("-----------------------------------")
    ##############################################################
    import skimage.io
    import skimage.transform

    img = skimage.io.imread('examples/model.png')
    img = skimage.transform.resize(img, (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))
    # print(img.shape)
    y_pred = blendmask(np.array([img, img]))
    outputs = ['sem_out', 'cls_logits', 'ctr_logits', 'reg_logits', 'mask_logits']
    # for y, name in zip(y_pred, outputs):
    #     print(name, y.shape)
    logits = [y_pred[1], y_pred[2], y_pred[3]]
    results = decode_predictions(logits)
    print(results)
