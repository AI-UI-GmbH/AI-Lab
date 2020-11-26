import os
import json

import numpy as np

import scipy
import skimage.draw
import skimage.color
import skimage.io
import skimage.transform


class Dataset(object):
    """The base class for dataset classes.

    """

    @property
    def num_classes(self):
        return len(self.classes)

    def __len__(self):
        return len(self.imgs)

    def __init__(self, data_dir, annotation_dir, cfg, classes=None):
        """

        Args:
            data_dir: directory of image files
            cfg:
            annotation_dir:
            classes: list of classes, background not included
        """
        # save annotations, path, filename, height, width
        self.data_dir = data_dir
        self.ann_dir = annotation_dir
        self.cfg = cfg
        self.imgs = []
        self.classes = ['BG']
        if classes:
            self.classes = classes

    def load_dataset(self):
        """VIA format annotation json data assumed
        """
        with open(self.ann_dir, 'rb') as f:
            annotations = json.load(f)

        for annotation in annotations.values():
            class_ids = []
            valid_annotations = []
            regions = annotation['regions']
            for ind, region in enumerate(regions.values()):
                region_attribute = region['region_attributes']
                shape_attribute = region['shape_attributes']

                # check if label exists
                label = region_attribute.get('label')
                if label is None:
                    continue
                # check if label is new
                if label not in self.classes:
                    self.classes.append(label)
                class_ids.append(self.classes.index(label))
                valid_annotations.append(shape_attribute)
            if 'width' not in annotation or 'height' not in annotation:
                filename = os.path.join(self.data_dir, annotation['filename'])
                img = skimage.io.imread(filename)
                annotation['height'] = img.shape[0]
                annotation['width'] = img.shape[1]
            self.add_image(
                filename=annotation['filename'],
                width=annotation['width'],
                height=annotation['height'],
                annotations=valid_annotations,
                class_ids=class_ids
            )

    def add_image(self, filename, width, height, annotations, class_ids, **kwargs):
        assert os.path.isfile(os.path.join(self.data_dir, filename)), filename + " doesn't exist"

        img_meta = {
            'filename': filename,
            'height': height,
            'width': width,
            'annotations': annotations,
            'class_ids': class_ids
        }
        img_meta.update(kwargs)
        self.imgs.append(img_meta)

    def load_image(self, img_id, channel_count):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        filename = self.imgs[img_id]['filename']
        filename = os.path.join(self.data_dir, filename)

        image = skimage.io.imread(filename)

        assert channel_count != 4 or image.ndim != 3, 'cannot transform rgba to rgb image'

        # Load image
        if channel_count == 1:
            if image.ndim == 3:  # convert rgb to gray
                image = skimage.color.rgb2gray(image).reshape(image.shape[0], image.shape[1], 1)
            elif image.ndim == 4:  # convert rgba to gray
                image = skimage.color.rgba2rgb(image).reshape(image.shape[0], image.shape[1], 1)
        elif channel_count == 3:
            if image.ndim != 3:  # convert gray to rgb
                image = skimage.color.gray2rgb(image)
            elif image.shape[-1] == 4:  # convert rgba to rgb
                image = image[..., :3]
        image_size = (self.cfg.DATA.IMAGE_SIZE, self.cfg.DATA.IMAGE_SIZE, image.shape[2])

        image = skimage.transform.resize(image, image_size)
        return image

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
        class_ids = img_meta['class_ids']

        masks = np.zeros([len(img_meta["annotations"]), img_meta["height"], img_meta["width"]],
                         dtype=np.uint8)
        for ind, sa in enumerate(img_meta["annotations"]):
            if sa['label'] == 'rect':
                rr, cc = skimage.draw.rectangle((sa['y'], sa['x']), (sa['height'] + sa['y'], sa['width'] + sa['x']))
            elif sa['label'] == 'polygon' or sa['label'] == 'polyline':
                rr, cc = skimage.draw.polygon(sa['all_points_y'], sa['all_points_x'])
            elif sa['label'] == 'ellipse':
                rr, cc = skimage.draw.ellipse(sa['ry'], sa['rx'], sa['cy'], sa['cx'])
            elif sa['label'] == 'circle':
                rr, cc = skimage.draw.circle(sa['cy'], sa['cx'], sa['r'])
            else:
                continue
            masks[ind, rr, cc] = 1
        scale_h, scale_w = self.cfg.DATA.IMAGE_SIZE/masks.shape[1], self.cfg.DATA.IMAGE_SIZE/masks.shape[2]
        masks = scipy.ndimage.zoom(masks, zoom=[1, scale_h, scale_w], order=0)
        class_ids = np.array(class_ids, dtype=np.int32)
        return masks.astype(np.bool), class_ids

    def load_semantic(self, img_id, masks=None, class_ids=None):
        if masks is None or class_ids is None:
            masks, class_ids = self.load_mask(img_id)

        semantic = np.zeros(masks.shape[1:], dtype=np.uint8)
        for ind, class_id in enumerate(class_ids):
            mask = masks[ind, :, :]
            semantic[mask == 1] = class_id
        return semantic
