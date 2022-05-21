'''
此文件包括所有辅助函数
'''
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import cv2
import os

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0

#用于可视化图片并保存
def vis_save(original_img, pred, save_path, save_name):
    blue   = [30,144,255] # aorta
    green  = [0,255,0]    # gallbladder
    red    = [255,0,0]    # left kidney
    cyan   = [0,255,255]  # right kidney
    pink   = [255,0,255]  # liver
    yellow = [255,255,0]  # pancreas
    purple = [128,0,255]  # spleen
    orange = [255,128,0]  # stomach
    # original_img = original_img * 255.0 #将归一化的图像还原
    original_img = original_img.astype(np.uint8) #不修改就报错，不知为何
    pred = pred.astype(np.uint8)
    original_img = cv2.cvtColor(original_img,cv2.COLOR_GRAY2BGR)
    pred = cv2.cvtColor(pred,cv2.COLOR_GRAY2BGR)
    original_img = np.where(pred==1, np.full_like(original_img, blue  ), original_img)
    original_img = np.where(pred==2, np.full_like(original_img, green ), original_img)
    original_img = np.where(pred==3, np.full_like(original_img, red   ), original_img)
    original_img = np.where(pred==4, np.full_like(original_img, cyan  ), original_img)
    original_img = np.where(pred==5, np.full_like(original_img, pink  ), original_img)
    original_img = np.where(pred==6, np.full_like(original_img, yellow), original_img)
    original_img = np.where(pred==7, np.full_like(original_img, purple), original_img)
    original_img = np.where(pred==8, np.full_like(original_img, orange), original_img)
    original_img = cv2.cvtColor(original_img,cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(save_path, save_name), original_img)