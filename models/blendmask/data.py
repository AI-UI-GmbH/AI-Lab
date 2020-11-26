import numpy as np
import tensorflow as tf

from utils import extract_bboxes, mold_image
from models.fcos import compute_centers, compute_targets


def build_data_generator(dataset, cfg, shuffle=True):
    def data_generator():

        img_idx = -1
        img_ids = np.arange(len(dataset))

        centers = compute_centers(cfg.DATA.IMAGE_SIZE, cfg.DATA.IMAGE_SIZE)

        class_ids_size = cfg.DATA.MAX_GT_INSTANCES
        bboxes_size = (cfg.DATA.MAX_GT_INSTANCES, 4)
        masks_size = (cfg.DATA.MAX_GT_INSTANCES, cfg.DATA.IMAGE_SIZE, cfg.DATA.IMAGE_SIZE)

        while True:
            try:
                # Increment index to pick next image. Shuffle if at the start of an epoch.
                img_idx = (img_idx + 1) % len(img_ids)
                if shuffle and img_idx == 0:
                    np.random.shuffle(img_ids)

                # Get GT bounding boxes and masks for image.
                img_id = img_ids[img_idx]
                image = dataset.load_image(img_id, 3)
                image = mold_image(image, cfg)
                _masks, _class_ids = dataset.load_mask(img_id)
                _bboxes = extract_bboxes(_masks)
                # If more instances than fits in the array, sub-sample from them.
                if _bboxes.shape[0] > cfg.DATA.MAX_GT_INSTANCES:
                    ids = np.random.choice(np.arange(_bboxes.shape[0]),
                                        cfg.DATA.MAX_GT_INSTANCES,
                                        replace=False)
                    _class_ids = _class_ids[ids]
                    _bboxes = _bboxes[ids]
                    _masks = _masks[ids, :, :]

                semantic = dataset.load_semantic(img_id, _masks, _class_ids)

                class_ids = np.zeros(class_ids_size, dtype=np.float32)
                bboxes = np.zeros(bboxes_size, dtype=np.float32)
                masks = np.zeros(masks_size, dtype=np.bool)

                class_ids[:_class_ids.shape[0]] = _class_ids
                bboxes[:_bboxes.shape[0]] = _bboxes
                masks[:_masks.shape[0], :, :] = _masks
                
                fcos_targets = compute_targets(class_ids, bboxes, centers)
                cls_targets, ctr_targets, reg_targets, valid_centers, normalizer_value, center_indices = fcos_targets
                # normalize bboxes
                bboxes = (bboxes - [0, 0, 1, 1]) / [masks.shape[1], masks.shape[2], masks.shape[1], masks.shape[2]]
                
                cls_targets = cls_targets.numpy()
                ctr_targets = ctr_targets.numpy()
                reg_targets = reg_targets.numpy()
                val_centers = valid_centers.numpy()
                normalizer_val = normalizer_value.numpy()
                center_indices = center_indices.numpy()

                yield image, semantic, cls_targets, ctr_targets, reg_targets, val_centers, \
                    normalizer_val, tf.concat(centers, axis=0), bboxes, masks, center_indices

            except (GeneratorExit, KeyboardInterrupt):
                raise
    return data_generator
