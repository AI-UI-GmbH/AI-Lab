import tensorflow as tf


def build_loss_fn(cfg):
    def loss(sem_targets, sem_logits):
        sem_targets = tf.expand_dims(sem_targets, axis=-1)
        sem_targets = tf.image.resize(sem_targets, sem_logits.shape[1:3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        sem_targets = tf.one_hot(sem_targets, depth=cfg.DATA.NUM_CLASSES)
        sem_targets = tf.squeeze(sem_targets)

        _loss = tf.losses.categorical_crossentropy(y_true=sem_targets, y_pred=sem_logits) * cfg.TRAIN.SEM_LOSS_WEIGHT
        _loss = tf.math.reduce_mean(_loss, axis=-1)
        _loss = tf.math.reduce_sum(_loss)
        # _loss = tf.nn.compute_average_loss(_loss, global_batch_size=cfg.TRAIN.BATCH_SIZE)
        return _loss

    return loss
