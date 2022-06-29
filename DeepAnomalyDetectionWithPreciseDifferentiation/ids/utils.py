import math
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from matplotlib.font_manager import FontProperties
from sklearn import metrics
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DataSetPath = "../dataset"
TrainDatasetFileName = "dataset_20220211_normal.csv"
TestDatasetFileName = "dataset_2_11_little.csv"
LogFilePath = "../log/log.txt"
TrainLogPath = "../log/TrainLog"
TrainResultPath = "../log/TrainLog"
TrainedModelPath = "../models"
TestLogPath = "../log/TestLog"
TestResultPath = "../log/TestLog"


def check_path(path):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        os.mkdir(path)
        print(f"Make new directory {path}")


def check_exists(path):
    return os.path.exists(path)


def read_param_config(file_name):
    with open(file_name) as f:
        param = yaml.load(f, Loader=yaml.FullLoader)
    return param


def get_parameters(config):
    ret = {}
    test_name = config['TestName']
    check_path(os.path.join(TrainLogPath, test_name))
    check_path(os.path.join(TrainResultPath, test_name))
    check_path(os.path.join(TestLogPath, test_name))
    check_path(os.path.join(TestResultPath, test_name))
    check_path(os.path.join(TrainedModelPath, test_name))
    ret['TrainLogPath'] = os.path.join(TrainLogPath, test_name)
    ret['TrainResultPath'] = os.path.join(TrainResultPath, test_name)
    ret['TestLogPath'] = os.path.join(TestLogPath, test_name)
    ret['TestResultPath'] = os.path.join(TestResultPath, test_name)
    ret['TrainedModelPath'] = os.path.join(TrainedModelPath, test_name)
    ret['TrainDatasetPath'] = os.path.join(DataSetPath, TrainDatasetFileName)
    ret['TestDatasetPath'] = os.path.join(DataSetPath, TestDatasetFileName)
    ret['Parameters'] = []
    default_param = config['DefaultParameter']
    for i, param in enumerate(config['Specific']):
        new_param = default_param.copy()
        new_param['Index'] = i
        for k, v in param.items():
            new_param[k] = v
        ret['Parameters'].append(new_param)
    return ret


def read_system_config(file_name):
    global DEVICE
    global DataSetPath
    global TrainDatasetFileName
    global TestDatasetFileName
    global LogFilePath
    global TrainLogPath
    global TrainResultPath
    global TrainedModelPath
    global TestLogPath
    global TestResultPath
    with open(file_name) as f:
        param = yaml.load(f, Loader=yaml.FullLoader)
    if param['Device'].upper() == 'GPU':
        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        DEVICE = torch.device('cpu')
    # Dataset setting
    DataSetPath = param['DataSetPath']
    check_path(DataSetPath)
    TrainDatasetFileName = param['TrainDatasetFileName']
    TestDatasetFileName = param['TestDatasetFileName']
    # Sys Log setting
    LogFilePath = param['LogFilePath']
    check_path(os.path.join(LogFilePath, "../"))
    # Train setting
    TrainLogPath = param['TrainLogPath']
    check_path(TrainLogPath)
    TrainResultPath = param['TrainResultPath']
    check_path(TrainResultPath)
    TrainedModelPath = param['TrainedModelPath']
    check_path(TrainedModelPath)
    # Test setting
    TestLogPath = param['TestLogPath']
    check_path(TestLogPath)
    TestResultPath = param['TestResultPath']
    check_path(TestResultPath)


def get_device():
    return DEVICE


def time_since(since, detail=False):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    if detail:
        return "%dm %fs" % (m, s)
    return "%dm %ds" % (m, s)


def create_tensor(tensor):  # 是否使用GPU
    return tensor.to(DEVICE)


def draw_effect(acc_list):
    # 画图
    epoch = np.arange(1, len(acc_list) + 1, 1)  # 步长为1
    acc_list = np.array(acc_list)
    plt.plot(epoch, acc_list)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid()  # 显示网格线 1=True=默认显示；0=False=不显示
    plt.show()


def draw_confusion_matrix(pred_lab, true_lab, class_lab, file_name=None):
    fonts = FontProperties(family="Times New Roman")
    conf_mat = confusion_matrix(true_lab, pred_lab)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=class_lab)
    disp.plot(cmap="YlGnBu", values_format='d')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if file_name is not None:
        if file_name != '':
            plt.savefig(file_name)
        else:
            plt.show()


def get_binary_class_rate(pred_label, true_label):
    tn, fp, fn, tp = confusion_matrix(true_label, pred_label).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    return tpr, tnr, fpr, fnr


def get_multiclass_rate(pred_label, true_label):
    mcm = multilabel_confusion_matrix(true_label, pred_label)
    tn = mcm[:, 0, 0]
    tp = mcm[:, 1, 1]
    fn = mcm[:, 1, 0]
    fp = mcm[:, 0, 1]
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    return tpr, tnr, fpr, fnr


def draw_roc_curve(preds, labels, title, file_name=None):
    fpr, tpr, thresholds = metrics.roc_curve(labels, preds)
    print(fpr,tpr)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=title)
    display.plot()
    if file_name is not None:
        if file_name != '':
            plt.savefig(file_name)
        else:
            plt.show()


def cal_auc(preds, labels):
    return metrics.roc_auc_score(labels, preds)
