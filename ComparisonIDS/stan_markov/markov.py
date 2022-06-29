import os.path
import sys
from collections import Counter

SYS_LOG = sys.stdout


def _check_file_exists(file):
    if os.path.exists(file) and not os.path.isdir(file):
        return True
    return False


class Markov:
    def __init__(self, para_file_path, is_to_train=False):
        self.occur = {}
        self.trans = {}
        self.state_prob = {}
        self.trans_prob = {}
        if _check_file_exists(para_file_path) and is_to_train is False:
            # Load parameters from files
            self.is_trained = self.__load_parameters_from_file(para_file_path)
        else:
            self.is_trained = False
        self.file_path = para_file_path

    def __load_parameters_from_file(self, file_path):
        return True

    def train(self, training_data=None):
        if training_data is None:
            print("No training data provided.", file=SYS_LOG, flush=True)
            return
        # state_i = [item[0] for item in training_data]
        # dataset_len = len(training_data)
        # self.occur = dict(Counter(state_i))
        # for k, v in self.occur.items():
        #     self.state_prob[k] = v / dataset_len
        #     state_i_list = {}
        #     for item in training_data:
        #         if item[0] == k:
        self.trans_prob = set(training_data)

    def match(self, input):
        if input in self.trans_prob:
            return True
        return False
