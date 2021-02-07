import cv2
import numpy as np

from torch.utils.data import DataLoader
import albumentations as A

from pipeline.datasets import VWPPEDataset
from utilities.common import collate_fn
from pipeline.transforms import ResizeWithKeepAspectRatio
from modules.lightning_module import LightningEquipmentDetNet
from configs.config_train_real import (backbone_cfg, loss_head_cfg, classes)

if __name__ == '__main__':
    colours = [(255, 0, 0), (255, 127, 0), (0, 0, 255),
               (255, 255, 0), (0, 255, 0), (46, 43, 95), (139, 0, 255)]
    idententy_per_class = dict(zip(range(len(classes)), list(zip(colours, classes))))
    path_to_dir = '/home/ivan/MLTasks/Datasets/ObjectDetection/PersonEquipmentTask/'
    checkpoint_path = '/home/ivan/MLTasks/home_projects/' \
                      'PersonalEquipmentDetection/results/real_epoch=29_mAP_05=0.4004.ckpt'
    width = 1088
    height = 640
    divider = 32

    transforms = A.Compose([
            ResizeWithKeepAspectRatio(height=height, width=width, divider=divider),
            A.Normalize(mean=(0., 0., 0.), std=(1., 1., 1.)),
            A.ToTensorV2()
        ])

    model = LightningEquipmentDetNet.load_from_checkpoint(checkpoint_path=checkpoint_path,
                                                          load_from_checkpoint=None,
                                                          backbone_cfg=backbone_cfg,
                                                          loss_head_cfg=loss_head_cfg)
    model.eval()
    model.cuda()

    dataset = VWPPEDataset(path_to_dir=path_to_dir,
                           mode='real_val',
                           transforms=transforms)
    dataloader = DataLoader(dataset,
                            collate_fn=collate_fn,
                            batch_size=1,
                            shuffle=False,
                            num_workers=1)
    thrs_conf = 0.6
    for batch in dataloader:
        prediction = model(batch['image'].cuda())
        gt_info = [{key: batch[key][info_idx] for key in list(batch.keys())}
                   for info_idx in range(len(batch['image']))]

        bboxes = model.loss_head.get_bboxes(prediction, gt_info)
        image = batch['image'][0]
        image = np.transpose(image.detach().cpu().numpy(), (1, 2, 0))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        bboxes = bboxes[0]['bboxes']

        for bbox in bboxes:
            x0y0 = (int(bbox[0]), int(bbox[1]))
            x1y1 = (int(bbox[2]), int(bbox[3]))
            if float(bbox[4] > thrs_conf):
                class_identeties = idententy_per_class[int(bbox[5])]
                image = cv2.rectangle(image, x0y0, x1y1, class_identeties[0], 3)
                image = cv2.putText(image, class_identeties[1], (x1y1[0], x0y0[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, class_identeties[0],
                                    2, cv2.LINE_AA)

        cv2.imshow('prediction', image)
        cv2.waitKey(0)




