import argparse
import scipy
from scipy import ndimage
import cv2
import numpy as np
import sys
import json
from tqdm import tqdm
import os
from PIL import Image as PILImage


PRED_DIR='E:/03_python_file/comp5421_TASK2/val_cv_result/'
GT_DIR='E:/03_python_file/comp5421_TASK2/val/labels/'
DATA_LIST_PATH='E:/03_python_file/comp5421_TASK2/val/val.txt'

NUM_CLASSES=7

parser = argparse.ArgumentParser(description="evaluate code")

parser.add_argument("--pred-dir", type=str, default=PRED_DIR,
                    help="Path to the directory containing prediction")
parser.add_argument("--gt-dir", type=str, default=GT_DIR,
                    help="Path to the directory containing groundtruth")
parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                    help="Path to the file listing the images in the dataset.")

args = parser.parse_args()


def main():
    confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
    image_list = [i_id.strip() for i_id in open(args.data_list)]
    tbar = tqdm(image_list)
    for index, img_name in enumerate(tbar):
        # print('%d/%d processd'%(index)%(len(testloader)))
        pred_file = os.path.join(args.pred_dir, "%s.png" % img_name)
        gt_file = os.path.join(args.gt_dir, "%s.png" % img_name)
        seg_pred = PILImage.open(pred_file)
        seg_gt   = PILImage.open(gt_file)
        seg_pred = np.array(seg_pred)
        seg_gt   = np.array(seg_gt)
        ignore_index = seg_gt != 255
        seg_gt = seg_gt[ignore_index]
        seg_pred = seg_pred[ignore_index]
        confusion_matrix += get_confusion_matrix(seg_gt, seg_pred, NUM_CLASSES)

        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)

        IU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IU = IU_array.mean()
        tbar.set_description('mIoU: %.4f' % (mean_IU))
    print({'meanIoU':mean_IU, 'IoU_array':IU_array})

def get_confusion_matrix(gt_label, pred_label, class_num):
        """
        Calcute the confusion matrix by given label and pred
        :param gt_label: the ground truth label
        :param pred_label: the pred label
        :param class_num: the nunber of class
        :return: the confusion matrix
        """
        index = (gt_label * class_num + pred_label).astype('int32')
        label_count = np.bincount(index)
        confusion_matrix = np.zeros((class_num, class_num))

        for i_label in range(class_num):
            for i_pred_label in range(class_num):
                cur_index = i_label * class_num + i_pred_label
                if cur_index < len(label_count):
                    confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

        return confusion_matrix

if __name__ == '__main__':
    main()
