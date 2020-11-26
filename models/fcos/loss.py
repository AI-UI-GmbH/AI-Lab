import tensorflow as tf


############################################################
#       loss function of FCOS
############################################################
def build_loss_fn(cfg):
    def loss(targets, logits):
        cls_targets, ctr_targets, reg_targets, valid_centers, normalizer_value, centers = targets
        cls_logits, ctr_logits, reg_logits = logits
        
        cls_loss = focal_loss(cls_targets, cls_logits, normalizer_value, cfg.DATA.NUM_CLASSES)
        ctr_loss = centerness_loss(ctr_targets, ctr_logits, valid_centers, normalizer_value)
        reg_loss = iou_loss(reg_targets, reg_logits, centers, valid_centers, normalizer_value)

        cls_loss = tf.nn.compute_average_loss(cls_loss, global_batch_size=cfg.DATA.BATCH_SIZE)
        ctr_loss = tf.nn.compute_average_loss(ctr_loss, global_batch_size=cfg.DATA.BATCH_SIZE)
        reg_loss = tf.nn.compute_average_loss(reg_loss, global_batch_size=cfg.DATA.BATCH_SIZE)
        return cls_loss, ctr_loss, reg_loss
    return loss


def focal_loss(cls_targets, cls_logits, normalizer_value, num_classes=10, alpha=0.25, gamma=2):
    cls_targets = tf.one_hot(tf.cast(cls_targets, dtype=tf.int32), depth=num_classes + 1)
    cls_targets = cls_targets[:, :, 1:]
    cls_logits = tf.sigmoid(cls_logits)

    at = alpha * cls_targets + (1 - cls_targets) * (1 - alpha)
    pt = cls_targets * cls_logits + (1 - cls_targets) * (1 - cls_logits)
    f_loss = at * tf.pow(1 - pt, gamma) * tf.nn.sigmoid_cross_entropy_with_logits(labels=cls_targets,
                                                                                  logits=cls_logits)
    f_loss = tf.reduce_sum(f_loss, axis=2)
    f_loss = tf.reduce_sum(f_loss, axis=1, keepdims=True)
    f_loss = f_loss / normalizer_value
    return f_loss


def centerness_loss(labels, logits, valid_centers, normalizer_value):
    labels = tf.reshape(labels, labels.shape)
    bce_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    bce_loss = bce_loss * tf.expand_dims(valid_centers, axis=-1)
    bce_loss = tf.reduce_sum(bce_loss, axis=1)
    bce_loss = bce_loss / normalizer_value
    return bce_loss


def iou_loss(reg_targets, reg_logits, centers, valid_centers, normalizer_value):
    boxes_true = tf.concat([centers - reg_targets[:, :, :2], centers + reg_targets[:, :, 2:]], axis=-1)
    boxes_pred = tf.concat([centers - reg_logits[:, :, :2], centers + reg_logits[:, :, 2:]], axis=-1)

    lu = tf.maximum(boxes_true[:, :, :2], boxes_pred[:, :, :2])
    rd = tf.minimum(boxes_true[:, :, 2:], boxes_pred[:, :, 2:])
    intersection = tf.maximum(0.0, rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    boxes_true_area = tf.reduce_prod(boxes_true[:, :, 2:] - boxes_true[:, :, :2], axis=2)
    boxes_pred_area = tf.reduce_prod(boxes_pred[:, :, 2:] - boxes_pred[:, :, :2], axis=2)
    union_area = tf.maximum(boxes_true_area + boxes_pred_area - intersection_area, 1e-8)
    iou = tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)
    
    # bg_mask = (1 - valid_centers) * 1e-8
    _iou_loss = iou + 1e-8
    _iou_loss = -1 * tf.math.log(_iou_loss)
    _iou_loss = _iou_loss * valid_centers
    _iou_loss = tf.reduce_sum(_iou_loss, axis=1, keepdims=True)
    _iou_loss = _iou_loss / normalizer_value
    return _iou_loss
