import tensorflow as tf
import tensorflow.keras.layers as KL

from models.fcos import fcos_head_graph
from models.fpn import fpn_graph
from models.protonet import protonet_graph


def BlendMask(cfg):
    """return BlendMask as keras model

    Arguments:
        cfg:
    """
    input_images = KL.Input(shape=[cfg.DATA.IMAGE_SIZE, cfg.DATA.IMAGE_SIZE, cfg.DATA.IMAGE_CHANNELS], name="input_images")
    # Conv2D layer to produce attention
    top_layer = KL.Conv2D(784, kernel_size=(3, 3), strides=(1, 1), padding='same', name='top_layer')

    fpn_outputs = fpn_graph(input_tensor=input_images,
                            pyramid_size=cfg.MODEL.FPN.PYRAMID_SIZE,
                            backbone=cfg.MODEL.BACKBONE,
                            extended_layer='P6P7')
    P2, P3, P4, P5, P6, P7 = fpn_outputs

    # proposal_generator: FCOS
    # INPUT: [P3, P4, P5, P6, P7]
    fcos_inputs = [P3, P4, P5, P6, P7]
    fcos_outputs = fcos_head_graph(inputs=fcos_inputs,
                                   cfg=cfg,
                                   top_layer=top_layer)
    cls_logits, reg_logits, ctr_logits, top_feats = fcos_outputs

    # basis module: ProtoNet
    # INPUT: [P3, P4, P5]
    protonet_inputs = [P3, P4, P5]
    protonet_outputs = protonet_graph(inputs=protonet_inputs, cfg=cfg)
    bases, sem_out = protonet_outputs

    # Model
    inputs = input_images
    # outputs = [sem_out, cls_logits, ctr_logits, reg_logits, mask_logits]
    outputs = [bases, sem_out, cls_logits, ctr_logits, reg_logits, top_feats]

    model = tf.keras.Model(inputs, outputs, name='blendmask')
    return model


############################################################
#       Blender
############################################################
# def blender_graph(inputs, cfg):
#     """
#
#     Args:
#         inputs:
#             1. bases
#             2. attentions with shape Nx(KxMxM)xHxW
#         cfg:
#
#     Returns:
#
#     """
#     assert cfg.ROI_RESOLUTION % cfg.ATTENTION_SIZE == 0, 'ROI_RESOLUTION and ATTENTION_SIZE must be divisible'
#
#     scale_ratio = int(cfg.ROI_RESOLUTION / cfg.ATTENTION_SIZE)
#
#     bases, attentions, reg_logits = inputs
#
#     rois = ROIAlign(cfg, name="roi_align_classifier")([bases, reg_logits])
#
#     mask_logits = KL.Lambda(lambda x: merge_bases(*x, cfg.ATTENTION_SIZE, scale_ratio))([rois, attentions])
#
#     return mask_logits
#
#
# def merge_bases(rois, coeffs, attention_size, scale_ratio):
#     # merge predictions
#     # N, B, H, W = rois.shape
#     coeffs = tf.concat(coeffs, axis=1)
#     N = tf.shape(rois)[0]
#     coeffs = tf.reshape(coeffs, (N, attention_size, attention_size, -1))
#
#     coeffs = KL.UpSampling2D(size=(scale_ratio, scale_ratio), interpolation="bilinear", name="blender_upsample")(coeffs)
#     coeffs = tf.keras.activations.softmax(coeffs, axis=1)
#     masks_logits = tf.reduce_sum((rois * coeffs), 1)
#
#     return tf.reshape(masks_logits, (N, -1))


def norm_boxes_graph(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [..., (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [..., (y1, x1, y2, x2)] in normalized coordinates
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.divide(boxes - shift, scale)
