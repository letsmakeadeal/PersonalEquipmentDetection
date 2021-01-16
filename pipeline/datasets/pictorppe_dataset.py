import os

from torch.utils.data import Dataset

import cv2
import copy
import numpy as np


class PictorPPEDataset(Dataset):
    def __init__(self,
                 path_to_dir: str,
                 mode: str = 'train',
                 debug: bool = False,
                 transforms=None):
        assert mode not in ['train', 'val', 'test'], 'Mode not in available modes'
        self.debug = debug
        self.mapping_to_vvppe_dataset = {0: 6, 1: 1, 2: 5}
        self._transforms = transforms
        path_to_labels = os.path.join(path_to_dir, 'Labels')
        path_to_images = os.path.join(path_to_dir, 'Images')
        if mode == 'train':
            ann_filename = 'pictor_ppe_crowdsourced_approach-02_train.txt'
        elif mode == 'val':
            ann_filename = 'pictor_ppe_crowdsourced_approach-02_valid.txt'
        elif mode == 'test':
            ann_filename = 'pictor_ppe_crowdsourced_approach-02_test.txt'

        path_to_anns = os.path.join(path_to_labels, ann_filename)
        self._images = []
        self._bboxes = []

        with open(path_to_anns) as file:
            for line in file.readlines():
                line_splitted = line.split(' ')
                image_filename = os.path.join(path_to_images, line_splitted[0])
                if not os.path.exists(image_filename):
                    continue
                bboxes_per_image = []
                for bbox in line_splitted[1:]:
                    x0, y0, x1, y1, class_id = map(int, bbox.split(','))
                    bboxes_per_image.append([x0, y0, x1, y1, class_id])

                if bboxes_per_image:
                    self._images.append(image_filename)
                    self._bboxes.append(bboxes_per_image)

        assert len(self._images) == len(self._bboxes), \
            "Annotations and images have not same length"

        print(f'Length of {mode} dataset is {len(self._images)}')

    def __getitem__(self, idx):
        filename = self._images[idx]
        bboxes = self._bboxes[idx]

        image = cv2.imread(filename)
        image_anno_dict = dict(image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                               bboxes=bboxes,
                               image_name=filename)

        if self._transforms is not None:
            image_anno_dict = self._transforms(**image_anno_dict)

        image_width = image_anno_dict['image'].shape[2]
        image_height = image_anno_dict['image'].shape[1]

        image_anno_dict['image_width'] = image_width
        image_anno_dict['image_height'] = image_height

        if self._debug and self._transforms is not None:
            image_copy_debug = image_anno_dict['image'].detach().cpu().numpy()
            image_copy_debug = np.transpose(image_copy_debug, (1, 2, 0))
            for bbox in image_anno_dict['bboxes']:
                x0y0 = (int(bbox[0]), int(bbox[1]))
                x1y1 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                image_copy_debug = cv2.rectangle(image_copy_debug, x0y0, x1y1, (0, 0, 255), 3)

            cv2.namedWindow('debug', cv2.WINDOW_KEEPRATIO)
            cv2.imshow('debug', image_copy_debug)
            cv2.waitKey(0)

        return image_anno_dict

