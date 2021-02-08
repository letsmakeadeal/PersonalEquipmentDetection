import os
import numpy as np

from torch.utils.data import Dataset

from pycocotools.coco import COCO
import cv2

from typing import List


class CocoDataset(Dataset):
    def __init__(self,
                 dataset_dir: str,
                 mode: str,
                 debug: bool,
                 transforms=None):
        super(CocoDataset, self).__init__()
        year = 2017
        self._debug = debug
        self._transforms = transforms
        self._image_dir = "{}/images/{}{}".format(dataset_dir, mode, year)
        self._coco = \
            COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, mode, year))
        self._class_ids = sorted(self._coco.getCatIds())
        self._image_ids = self._download_validated_images_anns()
        categories_names = [self._coco.loadCats(i)[0]["name"] for i in self._class_ids]
        self._class_id_to_category_name = dict(zip(self._class_ids, categories_names))
        self._class_id_plain_idx = dict(zip(self._class_ids, range(len(self._class_ids))))

        print(f'Length of COCO {mode} dataset is {len(self._image_ids)}')

    def _download_validated_images_anns(self):
        images_ids = list(self._coco.imgs.keys())
        imgs_idx_result = []
        for idx in range(len(images_ids)):
            idx_to_image_idx = images_ids[idx]
            anns = self._coco.loadAnns(self._coco.getAnnIds(
                imgIds=[idx_to_image_idx], catIds=self._class_ids, iscrowd=None))
            bboxes = [ann['bbox'] + [ann['category_id']] for ann in anns]
            if len(bboxes) == 0 or np.all([len(bbox) == 0 for bbox in bboxes]):
                continue
            imgs_idx_result.append(images_ids[idx])
        return imgs_idx_result

    def _prepare_bboxes_after_transforms(self,
                                         bboxes: List,
                                         images_width: int,
                                         images_height: int):
        output_bboxes = []
        for bbox in bboxes:
            if (bbox[1] < 0 and bbox[3] < 0) or \
                    (bbox[0] < 0 and bbox[2] < 0):
                continue
            bbox[0::2] = np.clip(bbox[0::2], 0, images_width)
            bbox[1::2] = np.clip(bbox[1::2], 0, images_height)

            output_bboxes.append([bbox[0], bbox[1], bbox[2], bbox[3],
                                  self._class_id_plain_idx[bbox[4]]])

        return output_bboxes

    def __getitem__(self, idx):
        idx_to_image_idx = self._image_ids[idx]
        anns = self._coco.loadAnns(self._coco.getAnnIds(
            imgIds=[idx_to_image_idx], catIds=self._class_ids, iscrowd=None))
        width = self._coco.imgs[idx_to_image_idx]["width"]
        height = self._coco.imgs[idx_to_image_idx]["height"]
        image_filename = os.path.join(self._image_dir,
                                      self._coco.imgs[idx_to_image_idx]['file_name'])
        bboxes = [ann['bbox'] + [ann['category_id']] for ann in anns]
        image = cv2.imread(image_filename)

        normalized_bboxes = [[float(bbox[0]) / width,
                              float(bbox[1]) / height,
                              float(bbox[2] + bbox[0]) / width,
                              float(bbox[3] + bbox[1]) / height,
                              bbox[4]] for bbox in bboxes]
        image_anno_dict = dict(image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                               bboxes=normalized_bboxes,
                               image_name=image_filename)

        if self._transforms is not None:
            image_anno_dict = self._transforms(**image_anno_dict)

        image_width = image_anno_dict['image'].shape[2]
        image_height = image_anno_dict['image'].shape[1]

        image_anno_dict['bboxes'] = [[box[0] * image_width, box[1] * image_height,
                                      box[2] * image_width, box[3] * image_height, box[4]]
                                     for box in image_anno_dict['bboxes']]

        image_anno_dict['image_width'] = image_width
        image_anno_dict['image_height'] = image_height

        if self._debug and self._transforms is not None:
            image_copy_debug = image_anno_dict['image'].detach().cpu().numpy()
            image_copy_debug = np.transpose(image_copy_debug, (1, 2, 0))
            for bbox in image_anno_dict['bboxes']:
                x0y0 = (int(bbox[0]), int(bbox[1]))
                x1y1 = (int(bbox[2]), int(bbox[3]))
                image_copy_debug = cv2.rectangle(image_copy_debug, x0y0, x1y1, (0, 0, 255), 3)
                image_copy_debug = cv2.putText(image_copy_debug, self._class_id_to_category_name[bbox[4]],

                                               x1y1, cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))
            cv2.imshow('debug', image_copy_debug)
            cv2.waitKey(0)

        image_anno_dict['bboxes'] = self._prepare_bboxes_after_transforms(
                                                    bboxes=image_anno_dict['bboxes'],
                                                    images_width=image_anno_dict['image_width'],
                                                    images_height=image_anno_dict['image_height'])
        return image_anno_dict

    def __len__(self):
        return len(self._image_ids)
