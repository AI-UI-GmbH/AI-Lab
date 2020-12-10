import numpy as np
import scipy

from pycocotools.coco import COCO

from config import config
from models import blendmask
from dataset import Dataset


class CocoDataset(Dataset):
    def __init__(self, data_dir, annotation_dir, cfg, classes=None):
        self.coco = COCO(annotation_dir)
        if classes is None:
            classes = ['BG'] + [cat['name'] for cat in list(self.coco.cats.values())]
        super().__init__(data_dir, annotation_dir, cfg, classes=classes)

        self.imgs = list(self.coco.imgs.values())
        for img in self.imgs: img['filename'] = img['file_name']

    def load_dataset(self):
        pass

    def load_mask(self, img_id):
        """Generate instance masks for an image.
        Args:
            img_id:
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        img_meta = self.imgs[img_id]
        img_id = img_meta['id']
        ann_ids = self.coco.getAnnIds(img_id)
        annotations = self.coco.loadAnns(ann_ids)

        class_ids = np.zeros(len(annotations), dtype=np.int32)
        masks = np.zeros([len(annotations), img_meta["height"], img_meta["width"]], dtype=np.uint8)

        for ind, annotation in enumerate(annotations):
            cat_id = annotation['category_id']
            cat_name = self.coco.cats[cat_id]['name']
            cls_id = self.classes.index(cat_name)
            class_ids[ind] = cls_id

            mask = self.coco.annToMask(annotation)
            masks[ind] = mask
        scale_h, scale_w = self.cfg.DATA.IMAGE_SIZE/masks.shape[1], self.cfg.DATA.IMAGE_SIZE/masks.shape[2]
        masks = scipy.ndimage.zoom(masks, zoom=[1, scale_h, scale_w], order=0)
        class_ids = np.array(class_ids, dtype=np.int32)
        return masks.astype(np.bool), class_ids


if __name__ == "__main__":
    config_filename = 'configs/blendmask.yaml'
    cfg = config(filename=config_filename)

    assert cfg.DATA.IMAGE_SIZE % 2 ** 6 == 0, "Image size must be dividable by 2 at least 6 times " \
                                              "to avoid fractions when downscaling and upscaling. " \
                                              "For example, use 256, 320, 384, 448, 512, ... etc."
    blendmask.train(cfg, CocoDataset)
