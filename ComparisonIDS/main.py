import csv
import time

import pandas as pd

from SECON.scheme import Secon
from stan_markov.scheme import Stan
from utils import evaluate_outputs, draw_confusion_matrix
import utils

TRAINING_DATA = ''

TEST_DATA = ''  


def add_label(cmd, attack):
    if attack == 0:
        return 0
    if cmd in [3108, 3138, 3169, 2178]:
        return 1  # 'Replay'
    if cmd in [8133, 12357]:
        return 2  # 'Forgery'
    if cmd in [7205, 9258, 13349]:
        return 3  # 'Dos'
    return 0


def preprocess_line(row: dict):
    for k, v in row.items():
        if v.startswith('0x'):
            row[k] = int(v, 16)
        else:
            row[k] = int(v)
    time_stamp = ((row['TimeHigh'] << 32) | row['TimeLow']) // 50  # 20ns
    del row['TimeHigh']
    del row['TimeLow']
    row['TimeStamp'] = time_stamp
    row['Attack'] = add_label(row['CMD1'], row['Attack'])
    return row


def bool_2_int(array_in):
    return [1 if i is False else 0 for i in array_in]


def eval_each_class(preds, labels, classes_labels=None):
    if classes_labels is None:
        classes_labels = ['Replay', 'Forgery', 'Dos']
    result = []
    ret = evaluate_outputs(preds, [0 if i == 0 else 1 for i in labels])
    print(ret)
    result.append(ret)

    for i, class_l in enumerate(classes_labels):
        index = i + 1
        p_list = []  # predict list
        t_list = []  # true list
        for j in range(len(labels)):
            if labels[j] == 0:
                p_list.append(preds[j])
                t_list.append(0)
            elif labels[j] == index:
                p_list.append(preds[j])
                t_list.append(1)
        ret = evaluate_outputs(p_list, t_list)
        print(f'Evaluation of class label {class_l} : {ret}')
        result.append(ret)
    pd.DataFrame(result).to_excel("Result.xlsx")


def stan_test(train_data, test_data, test_label):
    stan = Stan()
    stan.train(train_data, time_class_threshold=15)
    start = time.time()
    ret = stan.test(test_data, test_label)
    end = time.time()
    print(f'Test time {end - start}s in {len(test_label)} cases')
    # draw_confusion_matrix(bool_2_int(ret), bool_2_int(test_label), file_name="STAN_CONFUSION_MATRIX")
    eval_each_class(bool_2_int(ret), test_label)
    # utils.draw_roc_curve(bool_2_int(ret), [0 if i == 0 else 1 for i in labels], "Stan Roc Curve", 'STAN-ROC')


def secon_test(train_data, test_data, test_label):
    secon = Secon()
    secon.train(train_data)
    start = time.time()
    ret = secon.test(test_data, test_label)
    end = time.time()
    print(f'Test time {end - start}s in {len(test_label)} cases')
    print(evaluate_outputs(bool_2_int(ret), [0 if i == 0 else 1 for i in test_label]))
    # draw_confusion_matrix(bool_2_int(ret), bool_2_int(test_label), file_name="SECON_CONFUSION_MATRIX")
    eval_each_class(bool_2_int(ret), test_label)
    # utils.draw_roc_curve(bool_2_int(ret), [0 if i == 0 else 1 for i in labels], "SECON Roc Curve", 'SECON-ROC')


if __name__ == '__main__':
    raw_data = []
    with open(TRAINING_DATA, 'r') as f:
        f_csv = csv.DictReader(f)
        for row in f_csv:
            # preprocess data
            row = preprocess_line(row)
            raw_data.append(row)
    test_data = []
    labels = []
    with open(TEST_DATA, 'r') as f:
        f_csv = csv.DictReader(f)
        for row in f_csv:
            # preprocess data
            row = preprocess_line(row)
            test_data.append(row)
            # labels.append(True if row['Attack'] == 0 else False)
            labels.append(row['Attack'])
    if len(test_data) > 200204:
        test_data = test_data[:200204]
        labels = labels[:200204]
    # stan_test(raw_data, test_data, labels)
    secon_test(raw_data, test_data, labels)
