import enum
import sys
from enum import Enum

import numpy as np
import pandas as pd
import torch.utils.data
from ids.utils import get_device
import os

LOG_OUTPUT = sys.stdout


def set_log_output(output):
    global LOG_OUTPUT
    LOG_OUTPUT = output


class ReadType(Enum):
    CSV = enum.auto()


def _get_data_from_file(path, mode: ReadType = ReadType.CSV) -> pd.DataFrame:
    """
    Read raw data from the specific file.
    :param path:
    :param mode:
    :return:
    """
    if mode == ReadType.CSV:
        return pd.read_csv(path, on_bad_lines='warn')


class Dataset:
    raw_data_path: str
    preprocess_path: str
    preprocess_file_exist: bool
    data_len: int
    feature_num: int
    label_num: int
    tensor_dataset: torch.utils.data.TensorDataset

    preset_features = {
        *'CMD,Data1,Data2,Data3,Data4,Data5,Data6,Data7,Data8,Data9,Data10,Data11,Data12,Data13,Data14,Data15,Data16,Data17,Data18,Data19,Data20,Data21,Data22,Data23,Data24,Data25,Data26,Data27,Data28,Data29,Data30,Data31,Data32,STS,Label'.split(
            ','),
        *'RTAddr,TR,SubAddr,WdCnt,Data1,Data2,Data3,Data4,Data5,Data6,Data7,Data8,Data9,Data10,Data11,Data12,Data13,Data14,Data15,Data16,Data17,Data18,Data19,Data20,Data21,Data22,Data23,Data24,Data25,Data26,Data27,Data28,Data29,Data30,Data31,Data32,STS1,Label'.split(
            ','),
        'CMD1', 'CMD2', 'STS2'
    }

    def __init__(self, path: str, look_back=5, is_binary=False):
        # save parameters
        self.raw_data_path = path
        self.preprocess_path = path
        self.preprocess_file_exist = False
        self.label_encoder_file_exist = False
        self.look_back = look_back
        self.is_binary = is_binary

        # check path validation
        self._check_path()
        # get data
        # if preprocess data not exists or not to read preprocess data
        # preprocess data first
        if is_binary:
            self.label_classes = ['Benign', 'Anomaly']
            self.label_map = {'Benign': 0, 'Anomaly': 1}
            self.label_num = 2
        else:
            self.label_classes = ['Benign', 'Dos', 'Forgery', 'Replay']
            self.label_map = {'Benign': 0, 'Dos': 1, 'Forgery': 2, 'Replay': 3}
            self.label_num = 4
        self.feature_num = 0
        if not self.preprocess_file_exist:
            print(f"缺乏预构建的数据集，将使用原始数据重新构建:{os.path.basename(self.raw_data_path)}", file=LOG_OUTPUT)
            raw_data = _get_data_from_file(self.raw_data_path)
            data_shape = raw_data.shape
            print("原始数据集大小：", data_shape, file=LOG_OUTPUT)
            # data preprocess
            raw_data = self._raw_data_preprocess(raw_data)
            print("已完成原始数据集初始处理", file=LOG_OUTPUT)
            # separate features and labels
            raw_feature = raw_data.iloc[:, :-1]
            raw_label = raw_data.iloc[:, -1]
            self.feature_num = raw_feature.shape[1]
            # create dataset
            feature, labels = self._create_seq_dataset(raw_feature, raw_label)
            self.tensor_dataset = torch.utils.data.TensorDataset(feature, labels)
            self.data_len = len(labels)
            self._save_dataset()
        else:
            print(f"使用预构建的数据集：{os.path.basename(self.preprocess_path)}", file=LOG_OUTPUT)
            self._load_dataset()
            self.data_len, self.look_back, self.feature_num = self.tensor_dataset.tensors[0].size()
        self._conv_tensor_dataset()
        print(self, file=LOG_OUTPUT)

    def __repr__(self):
        info = f"""
原始数据集路径：{self.raw_data_path}
预构建数据集路径：{self.preprocess_path}
构建后数据集长度：{self.data_len}
构建后数据集序列长度（sequence length/look-back length）：{self.look_back}
构建后数据集特征个数：{self.feature_num}
构建后数据集标签种类个数：{self.label_num}
"""
        return f"Dataset({info})"

    def get_label_class(self, label_index):
        return self.label_classes[label_index]

    def get_label_classes(self):
        return self.label_classes

    def get_train_test_data(self, proportion=0.7, generator=None):
        train_len = int(proportion * self.data_len)
        test_len = self.data_len - train_len
        # train_len, test_len = int(proportion * self.data_len), self.data_len - int(proportion * self.data_len)
        if generator is None:
            return torch.utils.data.random_split(self.tensor_dataset, [train_len, test_len])
        else:
            return torch.utils.data.random_split(self.tensor_dataset, [train_len, test_len],
                                                 generator=torch.Generator().manual_seed(generator))

    def get_feature_num(self):
        return self.feature_num

    def get_label_num(self):
        return self.label_num

    def _conv_tensor_dataset(self):
        """
        convert features and labels in TensorDataset to the certain type.
        :return:
        """
        f, l = self.tensor_dataset.tensors
        f = f.to(get_device(), dtype=torch.float)  # turn to certain d-type and copy to certain device
        if self.is_binary:
            l = l.to(get_device(), dtype=torch.float)
        else:
            l = l.to(get_device(), dtype=torch.int64)
        self.tensor_dataset = torch.utils.data.TensorDataset(f, l)

    def _check_path(self):
        """
        Check whether path is valid. If file does not exist, program will exit.
        Only csv file is accepted. Suffix ".csv" is not necessary, function will auto-detect.
        :return:
        """
        print("Checking path...", end='', file=LOG_OUTPUT)
        abs_path = os.path.abspath(self.raw_data_path)
        dir_name, file_name = os.path.split(abs_path)
        if not file_name.endswith(".csv"):
            file_name += '.csv'
            abs_path = os.path.join(dir_name, file_name)
        self.raw_data_path = abs_path
        if not os.path.exists(abs_path) or not os.path.isfile(abs_path):
            print("Raw Dataset file does NOT exist. Exit with error.", file=LOG_OUTPUT)
            exit(1)
        if self.is_binary:
            preprocess_file_name = f'{file_name[:-4]}_preprocess_{self.look_back}_B.pt'
        else:
            preprocess_file_name = f'{file_name[:-4]}_preprocess_{self.look_back}.pt'
        # Put preprocessed data in a folder.
        preprocess_file_dir = os.path.join(dir_name, file_name.rstrip(".csv"))
        if not os.path.exists(preprocess_file_dir):
            os.mkdir(preprocess_file_dir)
        self.preprocess_path = os.path.join(preprocess_file_dir, preprocess_file_name)
        if os.path.exists(self.preprocess_path) and os.path.isfile(self.preprocess_path):
            self.preprocess_file_exist = True
        print("Success...", end='', file=LOG_OUTPUT)

    def _raw_data_preprocess(self, raw_data):
        # 对标签进行编码
        # 去掉不需要的列，也就是不在预设特征里的列
        raw_data = raw_data.drop(columns=list(set(raw_data) - self.preset_features))
        for col in raw_data.columns:
            if raw_data.dtypes[col] == "object":
                raw_data[col] = raw_data[col].fillna("NA")
            elif raw_data.dtypes[col] == "int64":
                raw_data[col] = raw_data[col].astype('float') / 65536
        return raw_data

    def _save_dataset(self):
        """
        save TensorDataset though torch api
        :return:
        """
        torch.save(self.tensor_dataset, self.preprocess_path)

    def _load_dataset(self):
        """
        read TensorDataset though torch api
        :return:
        """
        if self.preprocess_file_exist:
            self.tensor_dataset = torch.load(self.preprocess_path)

    def _create_seq_dataset(self, raw_feature, raw_label):
        sz = len(raw_label) - self.look_back
        data_feature, data_label = np.empty((sz, self.look_back, self.feature_num), dtype=np.float), \
                                   np.empty(sz, dtype=np.int)
        if isinstance(raw_feature, pd.DataFrame):
            raw_feature = raw_feature.to_numpy()
        if isinstance(raw_label, pd.Series):
            raw_label = raw_label.to_numpy()
        count = 1
        print("开始扩充序列", file=LOG_OUTPUT)
        for i in range(sz):
            data_feature[i] = (raw_feature[i:(i + self.look_back)])
            if self.is_binary:
                if raw_label[(i + self.look_back - 1)] == 'Benign':
                    data_label[i] = 0
                else:
                    data_label[i] = 1
            else:
                rlb = raw_label[(i + self.look_back - 1)]
                if rlb == 'Benign':
                    data_label[i] = 0
                elif rlb == 'Dos':
                    data_label[i] = 1
                elif rlb == 'Forgery':
                    data_label[i] = 2
                elif rlb == 'Replay':
                    data_label[i] = 3
            if i > count * 0.1 * sz:
                print('*' * 10, f"已经完成{count * 10}%", file=LOG_OUTPUT)
                count += 1
        print("序列扩充完毕", file=LOG_OUTPUT)
        feature = torch.from_numpy(data_feature)
        labels = torch.from_numpy(data_label)
        return feature, labels
