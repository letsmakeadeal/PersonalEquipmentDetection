from collections import defaultdict
from typing import Dict, List

import torch

from pytorch_lightning.metrics import Metric
from third_party_repositories.Object_Detection_Metrics import _init_paths
from third_party_repositories.Object_Detection_Metrics.lib.utils import *
from third_party_repositories.Object_Detection_Metrics.lib.BoundingBoxes import *
from third_party_repositories.Object_Detection_Metrics.lib.Evaluator import Evaluator

__all__ = ['mAP']


class mAP(Metric):
    def __init__(self,
                 labels_ids: Dict,
                 iou_treshold: float):
        super().__init__(compute_on_step=False)
        self._iou_treshold = iou_treshold
        self._ids = list(labels_ids.values())
        self._ids_names = list(labels_ids.keys())

        self._evaluator = Evaluator()
        self.add_state('_gt_bboxes', [])
        self.add_state('_pred_bboxes', [])

    def update(self, predicted_boxes, gt_info):
        for image_info in gt_info:
            for bbox_info in image_info['bboxes']:
                self._gt_bboxes.append(dict(x1=bbox_info[0].item(),
                                            y1=bbox_info[1].item(),
                                            x2=bbox_info[2].item(),
                                            y2=bbox_info[3].item(),
                                            class_id=bbox_info[4].item(),
                                            image_size=(image_info['image_width'],
                                                        image_info['image_height']),
                                            image_name=image_info['image_name']))

        for image_info in predicted_boxes:
            for bbox_info in image_info['bboxes']:
                self._pred_bboxes.append(dict(x1=bbox_info[0].item(),
                                              y1=bbox_info[1].item(),
                                              x2=bbox_info[2].item(),
                                              y2=bbox_info[3].item(),
                                              confidence=bbox_info[4].item(),
                                              class_id=bbox_info[5].item(),
                                              image_size=(image_info['image_width'],
                                                          image_info['image_height']),
                                              image_name=image_info['image_name']))

    def _get_evaluation_bboxes(self):
        all_bboxes = BoundingBoxes()
        for bbox in self._gt_bboxes:
            all_bboxes.addBoundingBox(BoundingBox(imageName=bbox['image_name'],
                                                  classId=bbox['class_id'],
                                                  x=bbox['x1'],
                                                  y=bbox['y1'],
                                                  w=bbox['x2'] - bbox['x1'],
                                                  h=bbox['y2'] - bbox['y1'],
                                                  typeCoordinates=CoordinatesType.Absolute,
                                                  classConfidence=1,
                                                  bbType=BBType.GroundTruth,
                                                  format=BBFormat.XYWH,
                                                  imgSize=bbox['image_size']))

        for bbox in self._pred_bboxes:
            all_bboxes.addBoundingBox(BoundingBox(imageName=bbox['image_name'],
                                                  classId=bbox['class_id'],
                                                  x=bbox['x1'],
                                                  y=bbox['y1'],
                                                  w=bbox['x2'] - bbox['x1'],
                                                  h=bbox['y2'] - bbox['y1'],
                                                  classConfidence=bbox['confidence'],
                                                  typeCoordinates=CoordinatesType.Absolute,
                                                  bbType=BBType.Detected,
                                                  format=BBFormat.XYWH,
                                                  imgSize=bbox['image_size']))
        return all_bboxes

    def compute(self):
        bounding_boxes = self._get_evaluation_bboxes()

        results = self._evaluator.PlotPrecisionRecallCurve(boundingBoxes=bounding_boxes,
                                                           IOUThreshold=self._iou_treshold,
                                                           showAP=False,
                                                           showInterpolatedPrecision=False,
                                                           showGraphic=False)

        return sum([result['AP'] for result in results]) / len(results)
