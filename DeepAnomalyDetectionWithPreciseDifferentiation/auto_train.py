import json
import math
import shutil
import sys
import time
import traceback
from json import JSONEncoder, JSONDecoder
from time import sleep
from typing import Any, Callable, List, Union

import numpy as np
import pandas as pd
from torch import nn
import os
import multiprocessing as mp

import torch
from torch.utils.data import DataLoader

from ids.generate_default_config import PARAM_CONFIG_FILE_NAME
from ids.traffic_classifier import BusClassifier
from ids.dataprocess import Dataset, set_log_output
from ids.models.gru_1553B_bus_classifier import BusClassifierGruNet
from ids.models.lstm_1553B_bus_classifier import BusClassifierLSTMNet
from ids.utils import create_tensor, time_since, read_param_config, get_parameters, LogFilePath


def load_parameter_from_file(file_name):
    file_name = os.path.abspath(file_name)
    if not file_name:
        return
    elif not os.path.exists(file_name):
        print("Config file not exists.")
    else:
        return read_param_config(file_name)


def auto_train(model_parameter: dict, output=sys.stdout):
    is_trained = False
    model_common_prefix = str(model_parameter['Index'])
    modal_file_name = model_common_prefix + '.pth'
    model_path = os.path.join(model_parameter['TrainedModelPath'], modal_file_name)
    if os.path.exists(model_path):
        print("Model already trained. Only test.", file=output, flush=True)
        is_trained = True
    train_log_path = os.path.join(model_parameter['TrainLogPath'], model_common_prefix + '.txt')
    train_result_path = os.path.join(model_parameter['TrainResultPath'], model_common_prefix + '.result')
    test_log_path = os.path.join(model_parameter['TestLogPath'], model_common_prefix + '.txt')
    test_result_path = os.path.join(model_parameter['TestResultPath'], model_common_prefix + '.result')
    if is_trained:
        f_train_log = sys.stdout
        f_train_result = sys.stdout
    else:
        f_train_log = open(train_log_path, 'w')
        f_train_result = open(train_result_path, 'w')
    f_test_log = open(test_log_path, 'w')
    f_test_result = open(test_result_path, 'w')
    try:
        # 数据读取
        # Use Class Dataset to preprocess raw data
        print("Reading Dataset......", file=output, end='', flush=True)
        dataset = Dataset(model_parameter['TrainDatasetPath'], look_back=model_parameter['SeqLen'])
        # dataset = Dataset(model_parameter['TrainDatasetPath'], look_back=model_parameter['SeqLen'], is_binary=True)
        class_labels = dataset.get_label_classes()
        train_set, test_set = dataset.get_train_test_data(proportion=0.8, generator=527)
        print("Success", file=output, flush=True)

        # Create DataLoader For Pytorch
        print("Generating DataLoader......", file=output, end='', flush=True)
        train_loader = DataLoader(train_set, batch_size=model_parameter['BatchSize'], num_workers=0, drop_last=True)
        test_loader = DataLoader(test_set, batch_size=model_parameter['BatchSize'], shuffle=True, num_workers=0,
                                 pin_memory=False)
        print("Success", file=output, flush=True)

        N_MESS = dataset.get_feature_num()
        N_TYPE = dataset.get_label_num()
        if N_TYPE == 2:
            N_TYPE = 1
        # Create Bus Classifier
        print("Defining Model......", file=output, end='', flush=True)
        model_type = model_parameter['TypeName']
        if "GRU" in model_type:
            classifier = BusClassifierGruNet(input_size=N_MESS, hidden_size=model_parameter['HiddenSize'],
                                             output_size=N_TYPE, batch_first=True, num_layers=model_parameter['Layer'],
                                             bidirectional=model_parameter['Bidirectional'],
                                             dropout=model_parameter['DropRate'])  # 定义模型
        elif model_type == "LSTM":
            classifier = BusClassifierLSTMNet(input_size=N_MESS, hidden_size=model_parameter['HiddenSize'],
                                              output_size=N_TYPE, batch_first=True, num_layers=model_parameter['Layer'],
                                              bidirectional=model_parameter['Bidirectional'],
                                              dropout=model_parameter['DropRate'])  # 定义模型
        else:
            print("Unsupported Model Type.", file=output, flush=True)
            return
        model_active_func = model_parameter['ActiveFunction']
        if model_active_func == 'ReLU':
            classifier.af = nn.ReLU()
        elif model_active_func == 'Tanh':
            classifier.af = nn.Tanh()
        elif model_active_func == 'Sigmoid':
            classifier.af = nn.Sigmoid()
        print("Success", file=output, flush=True)

        classifier = create_tensor(classifier)
        # 定义损失函数criterion，使用交叉熵损失函数
        print("Defining Loss Function......", file=output, end='', flush=True)
        if N_TYPE > 2:
            criterion = torch.nn.CrossEntropyLoss()
        else:
            criterion = torch.nn.BCEWithLogitsLoss()
        print("Success", file=output, flush=True)
        # 梯度下降使用的Adam算法
        print("Defining optimizer......", file=output, end='', flush=True)
        opt_choice = model_parameter['Optimizer']
        lr = model_parameter['LearningRate']
        if opt_choice == 'Adam':
            optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
        elif opt_choice == 'NAdam':
            optimizer = torch.optim.NAdam(classifier.parameters(), lr=lr)
        elif opt_choice == 'RMSprop':
            optimizer = torch.optim.RMSprop(classifier.parameters(), lr=lr)
        elif opt_choice == 'Adagrad':
            optimizer = torch.optim.Adagrad(classifier.parameters(), lr=lr)
        elif opt_choice == 'Adadelta':
            optimizer = torch.optim.Adadelta(classifier.parameters(), lr=lr)
        elif opt_choice == 'Adamax':
            optimizer = torch.optim.Adamax(classifier.parameters(), lr=lr)
        else:
            print(f'Wrong optimizer:{opt_choice}', file=output, flush=True)
            return
        print("Success", file=output, flush=True)
        bc = BusClassifier(classifier, num_epoch=model_parameter['EpochNum'], criterion=criterion, optimizer=optimizer,
                           model_path=model_path, train_log_output=f_train_log, train_result_output=f_train_result,
                           test_log_output=f_test_log, test_result_output=f_test_result)
        bc.set_class_labels(class_labels)
        bc.set_test_path(model_parameter['TestLogPath'])
        bc.set_model_name(model_common_prefix)

        train_result = None
        if not is_trained:
            print("Start training......", file=output, flush=True)
            train_result = bc.trainModel(train_loader)
            print("Success", file=output, flush=True)
        print("Start evaluating......", file=output, flush=True)
        test_result = bc.eval_model(test_loader)
        print("Success", file=output, flush=True)
    except Exception as e:
        print(e)
        traceback.print_exc()
        return
    finally:
        if not is_trained:
            f_train_log.close()
            f_train_result.close()
        f_test_log.close()
        f_test_result.close()
    return train_result, test_result


SAVE_FILE_NAME = 'Result.xlsx'


def save_results_to_file(results, is_binary=False):
    """
    :param results: param,train result,test result
    :return:
    """
    sz = len(results)

    r = []
    for param, train_result, test_result in results:
        res = test_result.copy()
        for k, v in param.items():
            if 'Train' in k: continue
            if 'Test' in k: continue
            if 'Log' in k: continue
            if 'Path' in k: continue
            if 'Result' in k: continue
            res[k] = v
        tc = res['TimeCost'].strip().split(' ')
        mnt = int(tc[0][:-1])
        sec = float(tc[1][:-1])
        tc = mnt * 60 + sec
        res['TimeCost'] = tc
        r.append(res)

    output = pd.DataFrame(r)
    output.to_excel(SAVE_FILE_NAME, index=False)


if __name__ == '__main__':
    config_file_name = os.path.basename(PARAM_CONFIG_FILE_NAME)
    config = load_parameter_from_file(config_file_name)
    parsed_param = get_parameters(config)
    # copy config file to train and test log
    shutil.copyfile(config_file_name, os.path.join(parsed_param['TrainLogPath'], config_file_name))
    shutil.copyfile(config_file_name, os.path.join(parsed_param['TestLogPath'], config_file_name))
    common_param = parsed_param.copy()
    del common_param['Parameters']
    print(parsed_param)
    with open(LogFilePath, 'w') as main_log:
        set_log_output(main_log)
        start = time.time()
        count = 0
        results = []
        for p in parsed_param['Parameters']:
            p.update(common_param)
            count += 1
            print(f'Now is {time_since(start)}', file=main_log)
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), file=main_log)
            print(f"No {count} Model is training.", file=main_log, flush=True)
            print(p, file=main_log, flush=True)
            ret = auto_train(p, output=main_log)
            if ret is not None:
                results.append((p, *ret))
        results.sort(key=lambda x: x[2]['Acc'], reverse=True)
        print('\n\n\n\n\n', file=main_log)
        save_results_to_file(results)
