import tensorflow as tf

from models.fcos import build_loss_fn as fcos_loss_fn
from models.protonet import build_loss_fn as protonet_loss_fn

from utils import roi_align


############################################################
#       loss function of blendmask
############################################################
def build_loss_fn(cfg):
    fcos_loss = fcos_loss_fn(cfg)
    protonet_loss = protonet_loss_fn(cfg)
    blender_loss = build_blender_loss_fn(cfg)

    def loss(y_true, y_pred):
        bases, sem_logits, cls_logits, ctr_logits, reg_logits, attentions = y_pred

        sem_targets, cls_targets, ctr_targets, reg_targets, valid_centers, \
            normalizer_value, centers, bbox_targets, mask_targets, center_indices = y_true

        cls_logits = tf.concat(cls_logits, axis=1)
        ctr_logits = tf.concat(ctr_logits, axis=1)
        reg_logits = tf.concat(reg_logits, axis=1)
        # FCOS losses
        fcos_targets = [cls_targets, ctr_targets, reg_targets, valid_centers, normalizer_value, centers]
        fcos_logits = [cls_logits, ctr_logits, reg_logits]
        cls_loss, ctr_loss, reg_loss = fcos_loss(fcos_targets, fcos_logits)

        # basis module loss
        sem_loss = protonet_loss(sem_targets, sem_logits)

        # blender loss
        attentions = tf.concat(attentions, axis=1)
        # print('bases: {0}, attentions: {1}, reg_targets: {2}, mask_targets: {3}'.format(bases.shape, attentions.shape, bbox_targets.shape, mask_targets.shape))
        _blender_loss = blender_loss(bases, attentions, ctr_targets, bbox_targets, mask_targets, normalizer_value, center_indices)

        return sem_loss, cls_loss, ctr_loss, reg_loss, _blender_loss

    return loss


############################################################
#       loss function of blender
############################################################
def build_blender_loss_fn(cfg):
    def blender_loss(bases, attentions, ctr_targets, reg_targets, mask_targets, loss_denorm, center_indices):
        
        assert cfg.MODEL.ROI_RESOLUTION % cfg.MODEL.ATTENTION_SIZE == 0, 'ROI_RESOLUTION and ATTENTION_SIZE must be divisible'

        scale_ratio = int(cfg.MODEL.ROI_RESOLUTION / cfg.MODEL.ATTENTION_SIZE)
        
        pool_shape = (cfg.MODEL.ROI_RESOLUTION, cfg.MODEL.ROI_RESOLUTION)
        rois = roi_align(bases, reg_targets, pool_shape)
        
        batch_size = center_indices.shape[0]
        
        rois = tf.reshape(rois, (batch_size, -1, rois.shape[1], rois.shape[2], rois.shape[3]))
        rois = tf.gather(rois, center_indices, batch_dims=-1, axis=1)
        rois = tf.reshape(rois, (-1, rois.shape[2], rois.shape[3], rois.shape[4]))

        mask_logits = merge_bases(rois, attentions, cfg.MODEL.ATTENTION_SIZE, scale_ratio)

        mask_targets = tf.reshape(tf.cast(mask_targets, tf.int32), (-1, mask_targets.shape[2], mask_targets.shape[3], 1))
        mask_targets = roi_align(mask_targets, reg_targets, pool_shape)
        # mask_targets = tf.reshape(mask_targets, (mask_targets.shape[0], -1))
        mask_targets = tf.reshape(mask_targets, (batch_size, -1, mask_targets.shape[1], mask_targets.shape[2], mask_targets.shape[3]))
        mask_targets = tf.gather(mask_targets, center_indices, batch_dims=-1, axis=1)
        mask_targets = tf.reshape(mask_targets, mask_logits.shape)

        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=mask_targets,
                                                       logits=mask_logits)

        loss = tf.reduce_sum(tf.reduce_mean(loss, axis=-1) * ctr_targets) / loss_denorm
        loss = tf.nn.compute_average_loss(loss, global_batch_size=cfg.DATA.BATCH_SIZE)
        return loss

    return blender_loss


def merge_bases(rois, coeffs, attention_size, scale_ratio):
    # merge predictions
    # N, B, H, W = rois.shape
    # coeffs = tf.concat(coeffs, axis=1)
    N = tf.shape(rois)[0]
    
    coeffs = tf.reshape(coeffs, (N, attention_size, attention_size, -1))
    coeffs = tf.keras.layers.UpSampling2D(size=(scale_ratio, scale_ratio), interpolation="bilinear")(coeffs)
    coeffs = tf.keras.activations.softmax(coeffs, axis=1)

    masks_logits = rois * coeffs
    masks_logits = tf.reduce_sum(masks_logits, -1)

    return tf.reshape(masks_logits, (N, -1))
