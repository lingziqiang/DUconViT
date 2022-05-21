import torch
import numpy as np
from tqdm import tqdm

def get_all_metrics(y_pred, y_true):
    results = []
    results.append(accuracy(y_pred, y_true))
    results.append(precision(y_pred, y_true))
    results.append(recall(y_pred, y_true))
    results.append(f1(y_pred, y_true))
    results.append(DSC(y_pred, y_true))
    results.append(HM(y_pred, y_true))
    results.append(IOU(y_pred, y_true))

    return results
def accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def precision(y_pred, y_true):
    tp, fp, _, _ = separate_count(y_pred, y_true)
    return divide(tp, tp + fp)


def recall(y_pred, y_true):
    tp, _, fn, _ = separate_count(y_pred, y_true)
    return divide(tp, tp + fn)


def f1(y_pred, y_true):
    pre = precision(y_pred, y_true)
    rec = recall(y_pred, y_true)
    return divide(2 * pre * rec, pre + rec)


def DSC(y_pred, y_true):
    return divide(2 * (y_true * y_pred).sum(), y_pred.sum() + y_true.sum())


def HM(y_pred, y_true):
    tp, fp, fn, _ = separate_count(y_pred, y_true)
    union = tp + fp + fn
    return divide(union - tp, union)


def IOU(y_pred, y_true):
    y_pred = (y_pred > 0.5)
    return divide((y_true * y_pred).sum(), ((y_true == 1) | (y_pred > 0.5)).sum())




def separate_count(y_pred, y_true):
    tp = np.sum((y_pred == y_true) & (y_true == 1))
    fp = np.sum((y_pred != y_true) & (y_true == 0))
    fn = np.sum((y_pred != y_true) & (y_true == 1))
    tn = np.sum((y_pred == y_true) & (y_true == 0))
    return tp, fp, fn, tn


def divide(dividend, divisor):
    return dividend / divisor if divisor != 0 else 0.0
