import tensorflow as tf

from .model import BlendMask
from .loss import build_loss_fn
from .data import build_data_generator


def build_model(cfg):
    model = BlendMask(cfg)
    optimizer = tf.keras.optimizers.SGD(learning_rate=cfg.TRAIN.OPTIMIZER.LEARNING_RATE,
                                        )
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


def summary(epoch, state, step, total_step, metrics, flush=True):
    print(f'Epoch {epoch + 1}: {step}/{total_step}', end=' ')
    for metric in metrics[:-1]:
        print(f'{metric.name}:', "{:.5f}".format(metric.result()), end=', ')
    print(f'{metrics[-1].name}:', "{:.5f}".format(metrics[-1].result()), end=' ')
    if flush:
        print(end='\r')
    else:
        print()


def train(cfg, dataset_cls):
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())
    with strategy.scope():
        model, loss_fn, optimizer, metrics = build_model(cfg)

        train_data, validation_data, train_data_size, validation_data_size = build_datasets(cfg, dataset_cls)
        train_data = strategy.experimental_distribute_dataset(train_data)
        validation_data = strategy.experimental_distribute_dataset(validation_data)
        checkpoint_path = f'trainings/training-{cfg.TRAIN.TRAINING}'
        model.load_weights(checkpoint_path + '/cp-12.ckpt')

    epoch = cfg.TRAIN.START_EPOCH
    train_steps = train_data_size // cfg.TRAIN.BATCH_SIZE
    validation_steps = validation_data_size // cfg.TRAIN.BATCH_SIZE

    with strategy.scope():
        def train_step(images, targets, step):
            with tf.GradientTape() as tape:
                logits = model(images)
                losses = loss_fn(targets, logits)
                total_loss = sum(losses)
            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return losses

        def test_step(images, targets, step):
            logits = model(images)
            losses = loss_fn(targets, logits)
            return losses

        def distributed_train_step(images, targets, step):
            per_replica_losses = strategy.experimental_run_v2(train_step, args=(images, targets, step,))
            # return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
            return per_replica_losses

        def distributed_test_step(images, targets, step):
            return strategy.experimental_run_v2(test_step, args=(images, targets, step,))

        # TRAIN
        """ for data, step in zip(train_data, range(2)):
            gradients, losses = grad(model, data, loss_fn)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            # UPDATE TRAIN RESULTS
            for metric, loss in zip(metrics[0::2], losses):
                metric(loss)
            summary(epoch, 'training', step, train_step, metrics[0::2]) """

        for _ in range(cfg.TRAIN.EPOCHS):
            for metric in metrics:
                metric.reset_states()
            print("epoch {0} starts".format(epoch))
            # TRAIN
            for data, step in zip(train_data, range(train_steps)):
                images = data[0]
                targets = data[1:]
                losses = distributed_train_step(images, targets, step)
                # UPDATE TRAIN RESULTS
                for metric, _loss in zip(metrics[0::2], losses):
                    metric(_loss)
                summary(epoch, 'training', step, train_steps, metrics[0::2])
            print(' ')
            # VALIDATION
            for data, step in zip(validation_data, range(validation_steps)):
                images = data[0]
                targets = data[1:]
                losses = distributed_test_step(images, targets, step)
                # UPDATE VALIDATION RESULTS
                for metric, _loss in zip(metrics[1::2], losses):
                    metric(_loss)
                summary(epoch, 'validating', step, validation_steps, metrics[1::2])
            summary(epoch, ' ', step, validation_steps, metrics[1::2], False)

            # SAVE MODEL
            checkpoint = checkpoint_path + f'/cp-{epoch}.ckpt'
            model.save_weights(checkpoint)
            print(f'Checkpoint saved: {checkpoint}.')
            epoch += 1

        model.save(checkpoint_path + '/model.h5')
        print(f'Training finished, model saved: {checkpoint_path}.')
