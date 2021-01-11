from torch import nn
from third_party_repositories.ScaledYOLOv4.models.models import Darknet


class YoloV4ScaledHead(nn.Module):
    def __init__(self):
        super(YoloV4ScaledHead, self).__init__()
        path_to_cfg = './third_party_repositories/ScaledYOLOv4/models/yolov4-csp.cfg'
        pass

    def forward(self):
        pass