from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import numpy as np
import torch
import runway
from runway.data_types import image, category, number
from constants import CATEGORIES

architectures = [
    'R-50-C4',
    'R-50-FPN',
    'R-101-FPN',
    'X-101-32x8d-FPN',
    'R-50-C4',
    'R-50-FPN',
    'R-101-FPN',
    'X-101-32x8d-FPN'
]


@runway.setup(options={'architecture': category(choices=architectures, default='R-50-FPN'), 'confidenceThreshold': number(min=0, max=1, step=0.1, default=0.7)})
def setup(opts):
    config_file = "configs/caffe2/e2e_mask_rcnn_%s_1x_caffe2.yaml" % opts['architecture'].replace(
        '-', '_')
    cfg.merge_from_file(config_file)
    if not torch.cuda.is_available():
        cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
    model = COCODemo(
        cfg,
        confidence_threshold=opts['confidenceThreshold'],
    )
    return model


@runway.command('mask', inputs={'image': image, 'category': category(choices=CATEGORIES, default='person')}, outputs={'output': image})
def mask(model, inputs):
    img = np.array(inputs['image'])
    out = model.run_on_opencv_image(img, inputs['category'])
    return {'output': out}


if __name__ == "__main__":
    runway.run()
