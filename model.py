from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import numpy as np
import torch
from runway import RunwayModel

masrcnn = RunwayModel()

@masrcnn.setup
def setup():
    config_file = "configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"

    cfg.merge_from_file(config_file)
    cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

    coco_demo = COCODemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.7,
    )
    return coco_demo

@masrcnn.command('mask', inputs={'image': 'image'}, outputs={'image': 'image'})
def mask(model, inp):
    img = np.array(inp['image'])
    output = coco_demo.run_on_opencv_image(img)
    return dict(image=output)

if __name__ == "__main__":
    masrcnn.run()