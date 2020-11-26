import tensorflow as tf
import tensorflow.keras.layers as KL

from utils import conv_block


def protonet_graph(inputs, cfg):
    P3, P4, P5 = inputs
    # refine
    R1 = conv_block(P3, 128)
    R2 = conv_block(P4, 128)
    R2 = KL.UpSampling2D(size=(2, 2), interpolation='bilinear', name="protonet_refine2upsampled")(R2)
    R3 = conv_block(P5, 128)
    R3 = KL.UpSampling2D(size=(4, 4), interpolation='bilinear', name="protonet_refine3upsampled")(R3)
    bases = KL.Add(name="protonet_add")([R1, R2, R3])
    # tower
    for _ in range(3):
        bases = conv_block(bases, 128)
    bases = KL.UpSampling2D(2, interpolation='bilinear')(bases)
    bases = conv_block(bases, 128)
    bases = KL.Conv2D(4, kernel_size=(1, 1), strides=(1, 1), name='bases_out')(bases)
    # seg_head
    sem_out = P3
    for _ in range(2):
        sem_out = conv_block(sem_out, 128)
    sem_out = KL.Conv2D(cfg.DATA.NUM_CLASSES, kernel_size=(1, 1), strides=(1, 1))(sem_out)
    return bases, sem_out


def ProtoNet(inputs, cfg):
    """

    """
    outputs = protonet_graph(inputs, cfg)

    model = tf.keras.Model(inputs, outputs, name='protonet')
    return model
