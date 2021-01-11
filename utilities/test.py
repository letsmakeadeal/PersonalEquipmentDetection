import cv2
import numpy as np

from torch.utils.data import DataLoader
import albumentations as A

from pipeline.dataset import VWPPEDataset
from utilities.common import collate_fn
from modules.lightning_module import LightningEquipmentDetNet
from config import (backbone_cfg, loss_head_cfg)

if __name__ == '__main__':
    path_to_dir = '/home/ivan/MLTasks/Datasets/ObjectDetection/PersonEquipmentTask'
    checkpoint_path = ''
    width = 544
    height = 320
    transforms = A.Compose([
            A.Resize(width=width, height=height),
            A.Normalize(mean=(0., 0., 0.), std=(1., 1., 1.)),
            A.ToTensorV2()
        ])

    model = LightningEquipmentDetNet.load_from_checkpoint(checkpoint_path=checkpoint_path,
                                                          backbone_cfg=backbone_cfg,
                                                          loss_head_cfg=loss_head_cfg)
    model.eval()
    model.cuda()

    dataset = VWPPEDataset(path_to_dir=path_to_dir,
                           is_train=False,
                           transforms=transforms)
    dataloader = DataLoader(dataset, collate_fn=collate_fn,
                            batch_size=1,
                            shuffle=False,
                            num_workers=1)

    for batch in dataloader:
        prediction = model(batch['image'])
        gt_info = [{key: batch[key][info_idx] for key in list(batch.keys())}
                   for info_idx in range(len(batch['image']))]

        bboxes = model.loss_head.get_bboxes(prediction, gt_info)
        image = batch['image'][0]
        image = np.transpose(image.detach.cpu().numpy(), (1, 2, 0)) * 255.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        bboxes = bboxes[0]

        for bbox in bboxes:
            x0y0 = (int(bbox[0]), int(bbox[1]))
            x1y1 = (int(bbox[2]), int(bbox[3]))
            image = cv2.rectangle(image, x0y0, x1y1, (0, 0, 255), 3)

        cv2.imshow('prediction', image)
        cv2.waitKey(0)




