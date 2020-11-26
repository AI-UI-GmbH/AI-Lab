import tensorflow as tf
import tensorflow.keras.layers as KL

from utils import conv_block


def FCOS(cfg, backbone=None, top_layer=None):
    """return FCOS as keras model

    Arguments:
        cfg: dict
        backbone: str or keras model
        top_layer: 

    Returns:
        keras model of FCOS
    """
    # init backbone
    # TODO
    inputs = backbone.outputs
    top_layer = None
    
    outputs = fcos_head_graph(inputs, cfg, top_layer)

    model = tf.keras.Model(inputs, outputs, name='fcos')
    return model


def fcos_head_graph(inputs, cfg, top_layer=None):
    """return FCOS graph

    Arguments:
        inputs: input tensors
        cfg: dict
        top_layer:
        p:

    Returns:
        output tensors of FCOS
    """
    # FCOS Head
    cls_logits = []
    reg_logits = []
    ctr_logits = []
    top_feats = []

    input_head1 = KL.Input(shape=[None, None, 256], name='cls_head')
    x = input_head1
    for _ in range(4):
        x = conv_block(x, filters=256)
    _cls_logits = KL.Conv2D(cfg.DATA.NUM_CLASSES, kernel_size=3, strides=1, padding="same", name="cls_logits")(x)
    _ctr_logits = KL.Conv2D(1, kernel_size=3, strides=1, padding="same", name="ctr_logits")(x)
    _cls_logits = KL.Reshape(target_shape=[-1, cfg.DATA.NUM_CLASSES], name="cls_logits_reshape")(_cls_logits)
    _ctr_logits = KL.Reshape(target_shape=[-1, 1], name="ctr_logits_reshape")(_ctr_logits)
    cls_head = tf.keras.Model(inputs=[input_head1], outputs=[_cls_logits, _ctr_logits], name='cls_head')

    input_head2 = KL.Input(shape=[None, None, 256], name='reg_head')
    x = input_head2
    for _ in range(4):
        x = conv_block(x, filters=256)
    _reg_logits = KL.Conv2D(4, kernel_size=3, strides=1, padding="same", name="reg_logits")(x)
    reg_head = tf.keras.Model(inputs=[input_head2], outputs=[_reg_logits], name='reg_head')

    for feature in inputs:
        cls_feature = cls_head(feature)
        reg_feature = reg_head(feature)

        cls_logits.append(cls_feature[0])
        ctr_logits.append(cls_feature[1])
        reg_logits.append(KL.Reshape(target_shape=[-1, 4])(reg_feature))

        if top_layer is not None:
            top_feat = top_layer(reg_feature)
            top_feat = KL.Reshape([-1, 784])(top_feat)
            top_feats.append(top_feat)

    # cls_logits = KL.Concatenate(axis=1)(cls_logits)
    # reg_logits = KL.Concatenate(axis=1)(reg_logits)
    # ctr_logits = KL.Concatenate(axis=1)(ctr_logits)
    # top_feats = KL.Concatenate(axis=1)(top_feats)

    return cls_logits, reg_logits, ctr_logits, top_feats
