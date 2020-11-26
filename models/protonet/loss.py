import tensorflow as tf


def build_loss_fn(cfg):
    def loss(sem_targets, sem_logits):
        sem_targets = tf.one_hot(sem_targets, depth=cfg.DATA.NUM_CLASSES)
        sem_targets = tf.image.resize(sem_targets, sem_logits.shape[1:3], method=tf.image.ResizeMethod.BILINEAR)
        
        _loss = tf.nn.softmax_cross_entropy_with_logits(sem_targets, sem_logits) * cfg.TRAIN.SEM_LOSS_WEIGHT
        _loss = tf.math.reduce_sum(_loss, -1)
        _loss = tf.math.reduce_sum(_loss, -1)
        _loss = tf.nn.compute_average_loss(_loss, global_batch_size=cfg.DATA.BATCH_SIZE)
        return _loss

    return loss
