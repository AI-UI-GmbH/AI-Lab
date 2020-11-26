# AI-Lab

AI-Lab is a toolbox open sourced by [Team AI-UI](https://www.ai4uandi.com) 
for image segmentation and reinforcement learning (later). AI-Lab is built upon
tensorflow 2.3. 

Now, AI-Lab implements:

- BlendMask

# What's Next

- Mask R-CNN
- Train and test on COCO dataset

# Quick Start

Change data folder and annotation folder in `configs/blendmask.yaml`

If not using our software [AI-UI](https://www.ai4uandi.com/download/) or [VIA](http://www.robots.ox.ac.uk/~vgg/software/via/)
as annotation tool, rewrite `load_mask` in `dataset.py` to adapt your annotation.

Run:
```
python train.py
```

# Train COCO dataset

Download COCO dataset and annotation files

Change data folder and annotation folder in `configs/blendmask.yaml`

Run:
```
python train_coco.py
```