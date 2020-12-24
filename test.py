from config import config
from models.mrcnn.model import mrcnn

if __name__ == '__main__':
    config_filename = 'configs/mask-rcnn.yaml'
    cfg = config(filename=config_filename)
    IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + cfg.DATA.NUM_CLASSES
    setattr(cfg.MODEL, 'NUM_CLASSES', cfg.DATA.NUM_CLASSES)
    setattr(cfg.MODEL, 'IMAGES_PER_GPU', cfg.TRAIN.IMAGES_PER_GPU)
    setattr(cfg.MODEL, 'BATCH_SIZE', cfg.TRAIN.BATCH_SIZE)
    setattr(cfg.MODEL, 'IMAGE_META_SIZE', IMAGE_META_SIZE)
    setattr(cfg.MODEL, 'IMAGE_SHAPE', [cfg.DATA.IMAGE_SIZE, cfg.DATA.IMAGE_SIZE, cfg.DATA.NUM_CLASSES])
    model = mrcnn(cfg.MODEL)
