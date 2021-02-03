import os

from torch.utils.data import Dataset

import copy
import numpy as np

import cv2
import imagesize


class PictorPPEDataset(Dataset):
    def __init__(self,
                 path_to_dir: str,
                 mode: str = 'train',
                 debug: bool = False,
                 transforms=None):
        assert mode in ['train', 'val', 'test'], 'Mode not in available modes'
        self._debug = debug
        self._labels_names = ['helmet', 'vest', 'person']
        self._mapping_to_vvppe_dataset = {0: 1, 1: 5, 2: 6}
        self._revert_mapping = dict(zip(self._mapping_to_vvppe_dataset.values(),
                                        self._mapping_to_vvppe_dataset.keys()))
        self._transforms = transforms
        self._path_to_dir = os.path.join(path_to_dir, "pictor-ppe")

        path_to_labels = os.path.join(self._path_to_dir, 'Labels')
        path_to_images = os.path.join(self._path_to_dir, 'Images')
        if mode == 'train':
            ann_filename = 'pictor_ppe_crowdsourced_approach-01_train.txt'
        elif mode == 'val':
            ann_filename = 'pictor_ppe_crowdsourced_approach-01_valid.txt'
        elif mode == 'test':
            ann_filename = 'pictor_ppe_crowdsourced_approach-01_test.txt'

        path_to_anns = os.path.join(path_to_labels, ann_filename)
        self._images = []
        self._bboxes = []

        with open(path_to_anns) as file:
            for line in file.readlines():
                line_splitted = line.split('\t')
                image_filename = os.path.join(path_to_images, line_splitted[0])
                if not os.path.exists(image_filename):
                    continue
                bboxes_per_image = []
                for bbox in line_splitted[1:]:
                    x0, y0, x1, y1, class_id = map(int, bbox.split(','))
                    class_id = self._mapping_to_vvppe_dataset[class_id]
                    bboxes_per_image.append([x0, y0, x1, y1, class_id])

                if bboxes_per_image:
                    self._images.append(image_filename)
                    self._bboxes.append(bboxes_per_image)

        assert len(self._images) == len(self._bboxes), \
            "Annotations and images have not same length"

        print(f'Length of PictorPPEDataset {mode} dataset is {len(self._images)}')

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        filename = self._images[idx]
        bboxes = self._bboxes[idx]
        image = cv2.imread(filename)
        
        # Normalizing bboxes because transforms needed it
        bboxes = [[float(box[0]) / image.shape[1], float(box[1]) / image.shape[0],
                   float(box[2]) / image.shape[1], float(box[3]) / image.shape[0],
                   box[4]] for box in bboxes]

        image_anno_dict = dict(image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                               bboxes=bboxes,
                               image_name=filename)

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
                image_copy_debug = cv2.putText(image_copy_debug, self._labels_names[self._revert_mapping[bbox[4]]],
                                               x1y1, cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))

            cv2.namedWindow('debug', cv2.WINDOW_KEEPRATIO)
            cv2.imshow('debug', image_copy_debug)
            cv2.waitKey(0)

        return image_anno_dict
