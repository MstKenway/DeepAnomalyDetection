from collections import Counter, defaultdict

from SECON.decision_tree import A_Detect
from SECON.period_model import P_Detect
from SECON.utils import is_multiple


class Secon:
    aperiodic_model: A_Detect

    def __init__(self):
        self.period_cmd = set()
        self.aperiodic_cmd = set()
        self.period_model = P_Detect()
        self.aperiodic_model = None

    @staticmethod
    def __check_multiple_period_cmd(period_list: list, cmd, period_num):
        idx = -1
        for i in range(len(period_list)):
            if cmd in period_list[i]:
                idx = i
                break
        if idx == -1:
            return False
        for i in range(idx, len(period_list), period_num):
            if cmd not in period_list[i]:
                return False
        return True

    def __classify_cmds(self, training_data):
        data_counter = Counter([item['CMD1'] for item in training_data])
        cmd_list = [item['CMD1'] for item in training_data]
        # 最大频数的命令字一定为周期性消息
        max_app_cmd, max_app_count = max(data_counter.items(), key=lambda x: x[1])
        if data_counter[training_data[0]['CMD1']] >= max_app_count - 1:
            max_app_cmd = training_data[0]['CMD1']
        period_list = []
        count = -1
        for cmd in cmd_list:
            if cmd == max_app_cmd:
                period_list.append([cmd])
                count += 1
            else:
                period_list[count].append(cmd)
        for k, v in data_counter.items():
            if max_app_count - 2 <= v <= max_app_count + 2:
                self.period_cmd.add(k)
            elif is_multiple(max_app_count, v) and self.__check_multiple_period_cmd(period_list, k,
                                                                                    round(max_app_count / v)):
                self.period_cmd.add(k)
            else:
                self.aperiodic_cmd.add(k)
        print(self.period_cmd)
        print(self.aperiodic_cmd)

    def train(self, training_data):
        self.__classify_cmds(training_data)
        # construct periodic training data
        p_data = []
        for row in training_data:
            cmd = row['CMD1']
            if cmd in self.period_cmd:
                p_data.append(row)
        self.period_model.train(p_data, self.period_cmd)
        # construct aperiodic training data
        a_data = {}
        a_statistic = set()
        # specify trigger message
        for idx in range(len(training_data)):
            cmd = training_data[idx]['CMD1']
            if cmd in self.aperiodic_cmd:
                a_statistic.add(training_data[idx - 1]['CMD1'])
        # init aperiodic models
        self.aperiodic_model = A_Detect(self.aperiodic_cmd, a_statistic)
        for item in a_statistic:
            a_data[item] = {'characters': [], 'labels': []}
        # collect trigger messages and its labels
        for idx in range(len(training_data) - 1):
            cmd = training_data[idx]['CMD1']
            if cmd in a_statistic:
                next_cmd = training_data[idx + 1]['CMD1']
                if next_cmd not in self.aperiodic_cmd:
                    next_cmd = -1
                a_data[cmd]['characters'].append(training_data[idx])
                a_data[cmd]['labels'].append(next_cmd)
        self.aperiodic_model.train(a_data)
        return a_data

    def test(self, test_data, labels):
        correct = 0
        count = len(labels)
        result = []
        self.period_model.clear()
        for i, row in enumerate(test_data):
            cmd = row['CMD1']
            time_stamp = row['TimeStamp']
            ret = self.period_model.detect_msg(cmd, time_stamp)
            if ret is False:
                ret = self.aperiodic_model.detect_msg(test_data[i - 1], cmd)
            result.append(ret)
        return result
