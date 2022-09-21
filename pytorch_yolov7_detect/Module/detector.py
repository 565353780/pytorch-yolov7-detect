#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append("../yolov7/")

import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from numpy import random

from models.common import Conv
from models.experimental import Ensemble
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.plots import plot_one_box


def attempt_load(weight, map_location=None):
    model = Ensemble()
    ckpt = torch.load(weight, map_location=map_location)  # load
    model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().
                 eval())  # FP32 model

    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set(
            )  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model


def letterbox(img,
              new_shape=(640, 640),
              color=(114, 114, 114),
              auto=True,
              scaleFill=False,
              scaleup=True,
              stride=32):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img,
                             top,
                             bottom,
                             left,
                             right,
                             cv2.BORDER_CONSTANT,
                             value=color)
    return img, ratio, (dw, dh)


class PytorchYoloV7Detector(object):

    def __init__(self, model_file_path=None):
        self.device = torch.device('cuda')
        self.imgsz = 640
        self.conf_thres = 0.25
        self.iou_thres = 0.45

        self.model = None
        self.names = None
        self.colors = None
        self.old_img_w = self.old_img_h = self.imgsz
        self.old_img_b = 1

        self.source = "/home/chli/chLi/NeRF/chair2/chair2.mp4"

        if model_file_path is not None:
            self.loadModel(model_file_path)
        return

    def loadModel(self, model_file_path):
        if not os.path.exists(model_file_path):
            print("[ERROR][PytorchYoloV7Detector::loadModel]")
            print("\t model_file not exist!")
            return False

        self.model = attempt_load(model_file_path, map_location=self.device)
        self.stride = int(self.model.stride.max())
        self.imgsz = check_img_size(self.imgsz, s=self.stride)
        self.model.half()

        self.names = model.module.names if hasattr(
            self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)]
                       for _ in self.names]

        self.model(
            torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(
                next(self.model.parameters())))
        return True

    @torch.no_grad()
    def detectImage(self, path):
        im0s = cv2.imread(path)
        img = letterbox(im0s, self.imgsz, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if (self.old_img_b != img.shape[0] or self.old_img_h != img.shape[2]
                or self.old_img_w != img.shape[3]):
            self.old_img_b = img.shape[0]
            self.old_img_h = img.shape[2]
            self.old_img_w = img.shape[3]
            for _ in range(3):
                self.model(img, augment=False)[0]

        pred = self.model(img, augment=False)[0]
        pred = non_max_suppression(pred,
                                   self.conf_thres,
                                   self.iou_thres,
                                   classes=None,
                                   agnostic=False)

        data = {'inputs': {}, 'predictions': {}, 'losses': {}, 'logs': {}}

        data['predictions']['results'] = []
        for det in pred:
            result_dict = {}
            gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4],
                                          im0s.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) /
                            gn).view(-1).tolist()  # normalized xywh
                    result_dict['class_id'] = cls
                    result_dict['conf'] = conf
                    result_dict['xywh'] = xywh
                    result_dict['label'] = self.names[int(cls)]

                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy,
                                 im0s,
                                 label=label,
                                 color=self.colors[int(cls)],
                                 line_thickness=1)
                data['predictions']['results'].append(result_dict)
        data['predictions']['render'] = im0s
        return data


def demo():
    model_file_path = "/home/chli/chLi/yolov7/yolov7-e6e.pt"
    image_file_path = "/home/chli/chLi/NeRF/chair2/images/image_10.png"

    pytorch_yolov7_detector = PytorchYoloV7Detector(model_file_path)
    data = pytorch_yolov7_detector.detectImage(image_file_path)
    print(data)
    return True


def demo_folder():
    model_file_path = "/home/chli/chLi/yolov7/yolov7-e6e.pt"
    image_folder_path = "/home/chli/chLi/NeRF/chair2/images/"

    new_image_folder_path = "/home/chli/chLi/NeRF/chair2/new_images/"
    os.makedirs(new_image_folder_path, exist_ok=True)

    pytorch_yolov7_detector = PytorchYoloV7Detector(model_file_path)

    image_file_name_list = os.listdir(image_folder_path)
    for image_file_name in tqdm(image_file_name_list):
        if image_file_name[-4:] not in ['.jpg', '.png']:
            continue

        image_file_path = image_folder_path + image_file_name
        data = pytorch_yolov7_detector.detectImage(image_file_path)

        image = cv2.imread(image_file_path)

        save_image = np.zeros(image.shape, dtype=np.uint8)
        save_image_size = save_image.shape
        for result_dict in data['predictions']['results']:
            label = result_dict['label']
            if label != "chair":
                continue
            xywh = result_dict['xywh']
            image_center_x = int(xywh[0] * save_image_size[1])
            image_center_y = int(xywh[1] * save_image_size[0])
            bbox_x_diff = int(xywh[2] * save_image_size[1] / 2.0)
            bbox_y_diff = int(xywh[3] * save_image_size[0] / 2.0)
            save_image[image_center_y - bbox_y_diff:image_center_y +
                       bbox_y_diff,
                       image_center_x - bbox_x_diff:image_center_x +
                       bbox_x_diff] = image[image_center_y -
                                            bbox_y_diff:image_center_y +
                                            bbox_y_diff, image_center_x -
                                            bbox_x_diff:image_center_x +
                                            bbox_x_diff]

        new_image_file_path = new_image_folder_path + image_file_name
        cv2.imwrite(new_image_file_path, save_image)
    return True
