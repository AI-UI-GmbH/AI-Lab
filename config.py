import yaml


class config:
    def __init__(self, cfg=None, filename=None):
        if filename:
            with open(filename, "r") as f:
                cfg = yaml.unsafe_load(f)

        for key, val in cfg.items():
            if isinstance(val, dict):
                val = config(cfg=val)
            setattr(self, key, val)
