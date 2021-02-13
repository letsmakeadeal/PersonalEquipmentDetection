import cv2
import numpy as np
import os

import torch
import albumentations as A

from pipeline.transforms import ResizeWithKeepAspectRatio
from modules.lightning_module import LightningEquipmentDetNet
from configs.config_train_real import (backbone_cfg, loss_head_cfg, classes)


def nms(boxes, scores, overlap=0.5, top_k=200):
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)
    idx = idx[-top_k:]
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    count = 0
    while idx.numel() > 0:
        i = idx[-1]
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h
        rem_areas = torch.index_select(area, 0, idx)
        union = (rem_areas - inter) + area[i]
        IoU = inter.float() / union.float()
        idx = idx[IoU.le(overlap)]

    return keep, count


def process_image(image_initial,
                  model,
                  transforms):
    image_initial = cv2.cvtColor(image_initial, cv2.COLOR_BGR2RGB)
    image_dict = dict(image=image_initial)
    image = transforms(**image_dict)['image']

    image_to_debug = np.transpose(image.detach().cpu().numpy(), (1, 2, 0))
    image_to_debug = cv2.cvtColor(image_to_debug, cv2.COLOR_RGB2BGR)

    prediction = model(image.cuda().unsqueeze(0))

    gt_info = [dict(image_width=image.shape[2],
                    image_height=image.shape[1],
                    image_name=filename)]

    bboxes = model.loss_head.get_bboxes(prediction, gt_info)
    scores = bboxes[0]['bboxes'][:, 4]
    bboxes = bboxes[0]['bboxes']

    keep, count = nms(bboxes, scores)
    keep_bboxes = keep[:count]
    bboxes = bboxes[keep_bboxes]

    return bboxes, image_to_debug


def show_bboxes(bboxes,
                debug_image,
                idententy_per_class):
    for bbox in bboxes:
        x0y0 = (int(bbox[0]), int(bbox[1]))
        x1y1 = (int(bbox[2]), int(bbox[3]))
        if float(bbox[4] > thrs_conf):
            class_identeties = idententy_per_class[int(bbox[5])]
            debug_image = cv2.rectangle(debug_image, x0y0, x1y1, class_identeties[0], 3)
            debug_image = cv2.putText(debug_image, class_identeties[1], (x1y1[0], x0y0[1] - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, class_identeties[0],
                                      2, cv2.LINE_AA)

    cv2.imshow('debug', debug_image)


if __name__ == '__main__':
    colours = [(255, 0, 0), (255, 127, 0), (0, 0, 255),
               (255, 255, 0), (0, 255, 0), (46, 43, 95), (139, 0, 255)]
    idententy_per_class = dict(zip(range(len(classes)), list(zip(colours, classes))))
    path_to_dir = '/home/ivan/MLTasks/Datasets/ObjectDetection/PersonEquipmentTask/Videos'#real_dataset/valid'
    checkpoint_path = '/home/ivan/MLTasks/home_projects/' \
                      'PersonalEquipmentDetection/results/real_epoch=51_mAP_05=0.5773_res34.ckpt'
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

    thrs_conf = 0.2
    run_on_videos = True

    cv2.namedWindow("debug", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("debug", 800, 600)

    if run_on_videos:
        for video in os.listdir(path_to_dir):
            print(f'Playing video {video}')
            filename = os.path.join(path_to_dir, video)
            vidcap = cv2.VideoCapture(filename)
            success, image_initial = vidcap.read()
            success = True
            while success:
                success, image_initial = vidcap.read()
                if not success or image_initial is None:
                    continue
                key = cv2.waitKey(4)
                if key == 27:
                    exit(0)
                if key == ord('n'):
                    break
                if key == ord('f'):
                    continue

                bboxes, image_to_debug = process_image(image_initial=image_initial,
                                                       model=model,
                                                       transforms=transforms)

                show_bboxes(bboxes=bboxes,
                            debug_image=image_to_debug,
                            idententy_per_class=idententy_per_class)
    else:
        for filename in os.listdir(path_to_dir):
            if filename.split('.')[-1] == 'txt':
                continue
            image_initial = cv2.imread(os.path.join(path_to_dir, filename))
            bboxes, image_to_debug = process_image(image_initial=image_initial,
                                                   model=model,
                                                   transforms=transforms)

            show_bboxes(bboxes=bboxes,
                        debug_image=image_to_debug,
                        idententy_per_class=idententy_per_class)
