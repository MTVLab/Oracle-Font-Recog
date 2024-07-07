'''
Description: 
Author: YuanJiang
Date: 2024-07-07 10:54:45
'''
import os
import sys
import json
import random
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics import precision_score, recall_score, f1_score

def cal_mean_std(imgs_path: str):
    img_h, img_w = 32, 32
    means, stdevs = [], []
    img_list = []
    imgs_path_list = [file for file in os.listdir(imgs_path) if file.endswith('.jpg')]
    for item in imgs_path_list:
        img = cv2.imread(os.path.join(imgs_path, item))
        img = cv2.resize(img, (img_w, img_h))
        img = img[:, :, :, np.newaxis]
        img_list.append(img)
    imgs = np.concatenate(img_list, axis=3)
    imgs = imgs.astype(np.float32) / 255.
    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # 拉成一行
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    # BGR --> RGB ， CV读取的需要转换，PIL读取的不用转换
    means.reverse()
    stdevs.reverse()
    return means, stdevs


def read_data(root: str):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []
    train_images_label = []
    every_class_num = []
    supported = [".jpg", ".JPG", ".png", ".PNG"]
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        images.sort()
        image_class = class_indices[cla]
        every_class_num.append(len(images))
        for img_path in images:
            train_images_path.append(img_path)
            train_images_label.append(image_class)
    print("{} images were found in the dataset.".format(sum(every_class_num)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."

    return train_images_path, train_images_label


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, boxes, labels = data
        # images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device), boxes.to(device))
        # pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.long().to(device))
        # loss = CB_loss(labels.to(device), pred, every_class_number, len(every_class_number), 'softmax')
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num,
            optimizer.param_groups[0]["lr"]
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)

    all_preds = []
    all_labels = []
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, boxes, labels = data
        # images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device),boxes.to(device))
        # pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        all_preds.extend(pred_classes.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.long().to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}, ".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num
        )
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, precision, recall, f1

class TGNNDataSet(Dataset):
    """自定义数据集"""
    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        data = {"image": [], "box": []}
        file_path = self.images_path[item].replace('.jpg', '.txt')
        image = cv2.imread(self.images_path[item])
        with open(file_path, 'r') as file:
            lines = file.read().splitlines()
            if len(lines) > 64:
                lines = lines[0: 64]
            for line in lines:
                box = [int(x) for x in line.split(' ')[0:] if x != '']
                assert len(box) == 4
                box_img = self.resize(image[box[1]:box[3], box[0]:box[2]])
                norm_box = np.array([x / image.shape[0] for x in box]).reshape(-1, 4)
                data['image'].append(box_img)
                data['box'].append(norm_box)

        image = np.concatenate(tuple(data['image']), axis=1)
        boxes = np.concatenate(tuple(data['box']), axis=0)

        image = np.pad(image, ((0, 0), (0, (4096 - image.shape[1])), (0, 0)), mode='constant', constant_values=0)
        boxes = np.pad(boxes, ((0, 64 - boxes.shape[0]), (0, 0)), mode='constant', constant_values=0)

        label = self.images_class[item]
        if self.transform is not None:
            image = self.transform(image)

        return image, torch.from_numpy(boxes).float(), label

    def resize(self, image, size=64):
        height, width = image.shape[0:2]
        ratio = float(size) / max(height, width)
        new_height, new_width = int(height * ratio), int(width * ratio)
        img = cv2.resize(image, (new_width, new_height))
        white_image = np.ones((size, size, 3), dtype=np.uint8) * 255
        white_image[(size - new_height) // 2:(size - new_height) // 2 + new_height,
        (size - new_width) // 2:(size - new_width) // 2 + new_width] = img
        return white_image

    @staticmethod
    def collate_fn(batch):
        images, postions, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        postions = torch.stack(postions, dim=0)
        labels = torch.as_tensor(labels)
        return images, postions, labels


class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


if __name__ == '__main__':
    root = './dataset_240609/train/'
    classes = os.listdir(root)
    mean_t = []
    std_t = []
    for cls in classes:
        file_path = os.path.join(root, cls)
        means, std = cal_mean_std(file_path)
        mean_t.append(means)
        std_t.append(std)
    mean_t = np.round(np.mean(np.array(mean_t).reshape(-1, 3), axis=0), decimals=3)
    std_t = np.round(np.mean(np.array(std_t).reshape(-1, 3), axis=0), decimals=3)
    print(mean_t, std_t)
    mean_std = []
    mean_std.append(mean_t.tolist())
    mean_std.append(std_t.tolist())
    print(mean_std)
    json_str = json.dumps(mean_std)
    with open('mean_std_train.json', 'w') as json_file:
        json_file.write(json_str)