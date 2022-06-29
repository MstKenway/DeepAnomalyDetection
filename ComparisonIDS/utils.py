import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def cal_acc(preds, labels):
    return metrics.accuracy_score(labels, preds)


def cal_pre_rec_fscore(preds, labels, beta=2.0):
    precision = metrics.precision_score(labels, preds)
    recall = metrics.recall_score(labels, preds)
    f_score = metrics.fbeta_score(labels, preds, beta=beta)
    return precision, recall, f_score


def cal_auc(preds, labels):
    return metrics.roc_auc_score(labels, preds)


def evaluate_outputs(preds, labels):
    ret = {}
    acc = cal_acc(preds, labels)
    precision, recall, f_score = cal_pre_rec_fscore(preds, labels)
    auc = cal_auc(preds, labels)
    tpr, tnr, fpr, fnr = cal_basic_indicator(preds, labels)
    ret['Acc'] = acc
    ret['Precision'] = precision
    ret['Recall'] = recall
    ret['F-Score'] = f_score
    ret['AUC'] = auc
    ret['Tpr'] = tpr
    ret['Tnr'] = tnr
    ret['Fpr'] = fpr
    ret['Fnr'] = fnr
    return ret


def draw_confusion_matrix(pred_lab, true_lab, class_label=None, file_name=None):
    if class_label is None:
        class_label = ['Benign', 'Anomaly']
    fonts = FontProperties(family="Times New Roman")
    conf_mat = confusion_matrix(true_lab, pred_lab)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=class_label)
    disp.plot(cmap="YlGnBu", values_format='d')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if file_name is not None:
        if file_name != '':
            plt.savefig(file_name)
        else:
            plt.show()


def cal_basic_indicator(preds, labels):
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    return tpr, tnr, fpr, fnr


def draw_roc_curve(preds, labels, title, file_name=None):
    fpr, tpr, thresholds = metrics.roc_curve(labels, preds)
    print(fpr, tpr)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=title)
    display.plot()
    if file_name is not None:
        if file_name != '':
            plt.savefig(file_name)
        else:
            plt.show()
