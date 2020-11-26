from config import config
from dataset import Dataset

from models import blendmask

if __name__ == "__main__":
    config_filename = 'configs/blendmask.yaml'
    cfg = config(filename=config_filename)

    assert cfg.DATA.IMAGE_SIZE % 2 ** 6 == 0, "Image size must be dividable by 2 at least 6 times " \
                                              "to avoid fractions when downscaling and upscaling. " \
                                              "For example, use 256, 320, 384, 448, 512, ... etc."
    blendmask.train(cfg, Dataset)
