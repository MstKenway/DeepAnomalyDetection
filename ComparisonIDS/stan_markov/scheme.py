from collections import Counter, defaultdict
import numpy as np

from stan_markov.markov import Markov
from stan_markov.rule import Rule
from stan_markov.state import State

PERIOD_MODEL_FILE_NAME = 'PeriodModel.txt'
APERIODIC_MODEL_FILE_NAME = 'AperiodicModel.txt'
MESSAGE_FILE_NAME = 'Messages.txt'


class Stan:
    def __init__(self):
        self.period_model = Markov(PERIOD_MODEL_FILE_NAME)
        self.aperiodic_model = Markov(APERIODIC_MODEL_FILE_NAME)
        self.p_rules = []
        self.a_rules = []

    def train(self, training_data, time_class_threshold=10, show_up_fre_pro=10):
        data_counter = Counter([item['CMD1'] for item in training_data])
        show_up_fre_line = max(data_counter.values()) // show_up_fre_pro

        msg_time_history = {}
        msg_time_statistic = defaultdict(list)
        period_set = []
        aperiodic_set = []
        # 1. Cal time cycle and collect
        for item in training_data:
            if msg_time_history.get(item['CMD1']) is None:
                msg_time_history[item['CMD1']] = item['TimeStamp']
                time_cycle = -1
            else:
                time_cycle = item['TimeStamp'] - msg_time_history[item['CMD1']]
                msg_time_history[item['CMD1']] = item['TimeStamp']
                msg_time_statistic[item['CMD1']].append(time_cycle)
            item['TimeCycle'] = time_cycle
        # 2. sort
        for k in msg_time_statistic.keys():
            msg_time_statistic[k].sort()
            last = msg_time_statistic[k][0]
            tc_list_all = [[]]
            tc_list = []
            classes = 0
            for v in msg_time_statistic[k]:
                if v - last <= 40:
                    tc_list_all[classes].append(v)
                else:
                    tc_list_all.append([v])
                last = v
            for tc in tc_list_all:
                tc_list.append(np.mean(tc))
            # judge if cmd is periodic
            if len(msg_time_statistic[k]) < show_up_fre_line or len(tc_list) >= time_class_threshold:
                aperiodic_set.append(k)
            else:
                period_set.append((k, tc_list))
        # print(period_set, aperiodic_set)

        # construct rules
        p_rules = []
        p_list = []
        a_rules = []
        a_list = []
        for i in period_set:
            p_rules.append(Rule(i[0], i[1], True))
            p_list.append(i[0])
        for i in aperiodic_set:
            a_rules.append(Rule(i, -1, False))
            a_list.append(i)
        self.p_rules = p_rules
        self.a_rules = a_rules
        print(p_list)
        print(a_list)
        # construct data set for markov
        p_dataset = []
        a_dataset = []
        last = None
        last_period = None
        for i, item in enumerate(training_data):
            cmd = item['CMD1']
            s = State(cmd)
            if i == 0:
                last_period = s
            else:
                if cmd in p_list:
                    p_dataset.append((last_period, s))
                    last_period = s
                else:
                    a_dataset.append((last, s))
            last = s
        self.period_model.train(p_dataset)
        self.aperiodic_model.train(a_dataset)

    def _judge(self, s, time_cycle):
        for i in self.p_rules:
            if i.match(s, time_cycle):
                return 0
        for i in self.a_rules:
            if i.match(s):
                return 1
        return -1

    def test(self, test_data, labels):
        msg_time_history = {}
        last = None
        last_period = None
        outputs = []
        for item in test_data:
            cmd = item['CMD1']
            if msg_time_history.get(cmd) is None:
                time_cycle = -1
            else:
                time_cycle = item['TimeStamp'] - msg_time_history[cmd]
            s = State(cmd)
            if last is None:
                last = s
                last_period = s
                outputs.append(True)
                continue
            ret = self._judge(s, time_cycle)
            if ret == -1:
                output = False
            elif ret == 0:
                output = self.period_model.match((last_period, s))
            else:
                output = self.aperiodic_model.match((last, s))
            outputs.append(output)
            if output is True:
                msg_time_history[item['CMD1']] = item['TimeStamp']
                if ret == 0:
                    last_period = s
                if ret != -1:
                    last = s
        return outputs
