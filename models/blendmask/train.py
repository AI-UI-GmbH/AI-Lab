import tensorflow as tf

from .model import BlendMask
from .loss import build_loss_fn
from .data import build_data_generator


def build_model(cfg):
    model = BlendMask(cfg)
    optimizer = tf.keras.optimizers.SGD(learning_rate=cfg.TRAIN.OPTIMIZER.LEARNING_RATE,
                                        clipvalue=cfg.TRAIN.OPTIMIZER.CLIP_VALUE)
    loss = build_loss_fn(cfg)

    metrics = []
    for metric_name in cfg.TRAIN.METRICS:
        name = metric_name + '_loss'
        metrics.append(tf.keras.metrics.Mean(name=name))
        name = 'val_' + name
        metrics.append(tf.keras.metrics.Mean(name=name))
    return model, loss, optimizer, metrics


def build_datasets(cfg, dataset_cls):
    train_set = dataset_cls(data_dir=cfg.DATA.TRAIN_DIR,
                            cfg=cfg,
                            annotation_dir=cfg.DATA.TRAIN_ANNOTATION)
    train_set.load_dataset()
    validation_set = dataset_cls(data_dir=cfg.DATA.VALIDATION_DIR,
                                 cfg=cfg,
                                 annotation_dir=cfg.DATA.VALIDATION_ANNOTATION,
                                 classes=train_set.classes)
    validation_set.load_dataset()
    assert train_set.classes == validation_set.classes, 'class information in train, validation dataset must be identical'

    data_types = (tf.float32, tf.int32, tf.float32, tf.float32, tf.float32,
                  tf.float32, tf.float32, tf.float32, tf.float32, tf.bool, tf.int32)

    train_data = tf.data.Dataset.from_generator(build_data_generator(train_set, cfg), data_types)
    train_data = train_data.batch(cfg.TRAIN.BATCH_SIZE)
    validation_data = tf.data.Dataset.from_generator(build_data_generator(validation_set, cfg), data_types)
    validation_data = validation_data.batch(cfg.TRAIN.BATCH_SIZE)
    return train_data, validation_data, len(train_set), len(validation_set)


def grad(model, data, loss):
    with tf.GradientTape() as tape:
        logits = model(data[0])
        losses = loss(data[1:], logits)
        total_loss = sum(losses)
    gradients = tape.gradient(total_loss, model.trainable_variables)
    return gradients, losses


def summary(epoch, state, step, total_step, metrics):
    print(f'Epoch {epoch + 1}: {state} {step}/{total_step}', end=' ')
    for metric in metrics:
        print(f'{metric.name}: {metric.result()}', end=', ')
    print(end='\r')


def train(cfg, dataset_cls):
    model, loss_fn, optimizer, metrics = build_model(cfg)

    train_data, validation_data, train_data_size, validation_data_size = build_datasets(cfg, dataset_cls)

    checkpoint_path = f'trainings/training-{cfg.TRAIN.TRAINING}'

    epoch = cfg.TRAIN.START_EPOCH
    train_step = train_data_size // cfg.TRAIN.BATCH_SIZE
    validation_step = validation_data_size // cfg.TRAIN.BATCH_SIZE

    for _ in range(cfg.TRAIN.EPOCHS):
        for metric in metrics:
            metric.reset_states()
        print("epoch {0} starts".format(epoch))
        # TRAIN
        for data, step in zip(train_data, range(train_step)):
            gradients, losses = grad(model, data, loss_fn, cfg.TRAIN.BATCH_SIZE)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            # UPDATE TRAIN RESULTS
            for metric, loss in zip(metrics[0::2], losses):
                metric(loss)
            summary(epoch, 'training', step, train_step, metrics[0::2])
        # VALIDATION
        for data, step in zip(validation_data, range(validation_step)):
            logits = model(data[0])
            losses = loss_fn(data[1:], logits)
            # UPDATE VALIDATION RESULTS
            for metric, loss in zip(metrics[1::2], losses):
                metric(loss)
            summary(epoch, 'training', step, train_step, metrics[1::2])

        # SAVE MODEL
        checkpoint = checkpoint_path + f'/cp-{epoch}'
        model.save_weights(checkpoint)
        print(f'Checkpoint saved: {checkpoint}.')
        epoch += 1

    model.save(checkpoint_path + '/model.h5')
    print(f'Training finished, model saved: {checkpoint_path}.')
