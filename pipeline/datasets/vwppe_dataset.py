import os

from torch.utils.data import Dataset

import cv2
import copy
import numpy as np


class VWPPEDataset(Dataset):
    def __init__(self,
                 path_to_dir: str,
                 mode: str,
                 debug: bool = False,
                 transforms=None):
        super(VWPPEDataset, self).__init__()
        assert mode in ['virtual_train', 'virtual_val',
                        'real_train', 'real_val']
        self._debug = debug
        self._transforms = transforms
        self._images = []
        self._anns = []
        self._mode = mode
        if mode == 'virtual_train' or mode == 'virtual_val':
            path_to_data = os.path.join(path_to_dir, "dataset_1088x612")
        elif mode == 'real_train':
            path_to_data = os.path.join("real_dataset", "train")
        elif mode == 'real_val':
            path_to_data = os.path.join("real_dataset", "valid")

        top_folder_dirs = os.listdir(path_to_data)
        if mode == 'virtual_train':
            working_folders = top_folder_dirs[:int(0.9*len(top_folder_dirs))]
        elif mode == 'virtual_val':
            working_folders = top_folder_dirs[int(0.9*len(top_folder_dirs)):]

        for subdir, dirs, files in os.walk(path_to_data):
            if any(folder in subdir for folder in working_folders):
                for file in files:
                    if 'txt' not in file.split('/')[-1]:
                        self._images.append(os.path.join(subdir, file))

        images_copy = copy.deepcopy(self._images)
        for image in images_copy:
            ann_filename = image.split('.')[0] + '.txt'
            if os.path.exists(ann_filename):
                if os.path.getsize(ann_filename) != 0:
                    self._anns.append(ann_filename)
                else:
                    self._images.remove(image)
            else:
                self._images.remove(image)

        assert len(self._images) == len(self._anns), \
            "Annotations and images have not same length"

        print(f'Length of {mode} dataset is {len(self._anns)}')

    @staticmethod
    def _get_annotations_from_file(filename: str):
        image_anno_dict = dict(image_name=filename,
                               bboxes=[])
        with open(filename) as file:
            for line in file.readlines():
                class_id, x0, y0, w, h = \
                    [float(value) for value in line.split(' ') if value != '\n']
                image_anno_dict['bboxes'].append([x0 - w / 2, y0 - h / 2, w, h, class_id])

        return image_anno_dict

    def __len__(self):
        return len(self._anns)

    def __getitem__(self, idx):
        image = cv2.imread(self._images[idx])
        image_anno_dict = self._get_annotations_from_file(self._anns[idx])
        image_anno_dict['image'] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self._transforms is not None:
            image_anno_dict = self._transforms(**image_anno_dict)

        image_width = image_anno_dict['image'].shape[2]
        image_height = image_anno_dict['image'].shape[1]
        image_anno_dict['bboxes'] = [[box[0] * image_width, box[1] * image_height,
                                      box[2] * image_width, box[3] * image_height, box[4]]
                                     for box in image_anno_dict['bboxes']]

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

        image_anno_dict['image_width'] = image_width
        image_anno_dict['image_height'] = image_height

        return image_anno_dict


