import tensorflow as tf


def compute_centers(height, width):
    centers = []
    for level in range(3, 8):
        centers.append(compute_centers_per_level(level, height, width))
    return centers


def compute_centers_per_level(level, height, width):
    stride = tf.cast(tf.pow(2, level), dtype=tf.float32)
    fm_h = tf.math.ceil(height / stride)
    fm_w = tf.math.ceil(width / stride)
    x = (tf.range(fm_w, dtype=tf.float32) + 0.5) * stride
    y = (tf.range(fm_h, dtype=tf.float32) + 0.5) * stride
    xx, yy = tf.meshgrid(x, y)
    xy = tf.stack([xx, yy], axis=-1)
    xy = tf.reshape(xy, shape=[-1, 2])
    return xy


def compute_level_target(boxes, class_ids, min_size, max_size, centers):
    centers = tf.cast(centers, tf.float32)
    boxes = tf.cast(boxes, tf.float32)

    xy_min = boxes[:, :2]
    xy_max = boxes[:, 2:]

    lt = tf.expand_dims(centers, axis=1) - xy_min
    rb = xy_max - tf.expand_dims(centers, axis=1)
    ltrb = tf.concat([lt, rb], axis=2)

    max_ltrb = tf.reduce_max(ltrb, axis=2)

    valid_size = tf.logical_and(tf.greater(max_ltrb, min_size), tf.less(max_ltrb, max_size))

    inside_box = tf.cast(tf.greater(ltrb, 0), dtype=tf.float32)
    inside_box = tf.not_equal(tf.reduce_prod(inside_box, axis=2), 0.)

    valid_boxes = tf.logical_and(inside_box, valid_size)
    valid_boxes = tf.cast(valid_boxes, dtype=tf.float32)

    valid_centers = tf.not_equal(tf.reduce_sum(valid_boxes, axis=1), 0)
    valid_centers = tf.cast(valid_centers, dtype=tf.float32)

    box_indices = tf.argmax(valid_boxes, axis=1)

    matched_boxes = tf.gather(boxes, box_indices)
    matched_class_ids = tf.reshape(tf.gather(class_ids, box_indices) + 1, (-1,))

    x_min, y_min, x_max, y_max = tf.split(matched_boxes, num_or_size_splits=4, axis=1)
    l = tf.abs(centers[:, 0] - x_min[:, 0])
    t = tf.abs(centers[:, 1] - y_min[:, 0])
    r = tf.abs(x_max[:, 0] - centers[:, 0])
    b = tf.abs(y_max[:, 0] - centers[:, 1])
    lr = tf.stack([l, r], axis=1)
    tb = tf.stack([t, b], axis=1)

    min_lr = tf.reduce_min(lr, axis=1)
    max_lr = tf.reduce_max(lr, axis=1)
    min_tb = tf.reduce_min(tb, axis=1)
    max_tb = tf.reduce_max(tb, axis=1)

    cls_target = matched_class_ids * valid_centers
    ctr_target = tf.sqrt((min_lr / max_lr) * (min_tb / max_tb)) * valid_centers
    ctr_target = tf.reshape(ctr_target, shape=[-1, 1])
    reg_target = tf.stack([l, t, r, b], axis=1) * tf.expand_dims(valid_centers, axis=1)

    return cls_target, ctr_target, reg_target, valid_centers, box_indices


def compute_targets(class_ids, boxes, centers):
    # areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    # indices = tf.argsort(areas, direction='ASCENDING')
    # boxes = tf.gather(boxes, indices=indices)
    # class_ids = tf.gather(class_ids, indices=indices)

    soi = [0] + [2 ** i * 64 for i in range(6)] + [1e8]
    targets = []
    for i in range(5):
        level_targets = compute_level_target(boxes, class_ids, soi[i], soi[i + 1], centers[i])
        targets.append(level_targets)

    cls_targets = tf.concat([target[0] for target in targets], axis=0)
    ctr_targets = tf.concat([target[1] for target in targets], axis=0)
    reg_targets = tf.concat([target[2] for target in targets], axis=0)
    valid_centers = tf.concat([target[3] for target in targets], axis=0)
    box_indices = tf.concat([target[4] for target in targets], axis=0)
    normalizer_value = tf.maximum(tf.reduce_sum(valid_centers, keepdims=True), 1.0)

    return cls_targets, ctr_targets, reg_targets, valid_centers, normalizer_value, box_indices
