from collections import Counter

from sklearn import tree


class A_Detect:
    def __init__(self, aperiodic_cmds, trigger_cmds):
        self.a_cmds = aperiodic_cmds
        self.t_cmds = trigger_cmds
        self.df = {}
        for i in trigger_cmds:
            if i not in [12357, 14405, 10312]:
                self.df[i] = tree.DecisionTreeClassifier(class_weight='balanced')
            else:
                self.df[i] = tree.DecisionTreeClassifier(class_weight={-1: 0.05, 2114: 0.95})

    @staticmethod
    def __prepross_data(cmd, rows):
        data_name_list = [f'Data{i}' for i in range(1, 33)]
        word_count = cmd & 0x1f
        if word_count == 0:
            word_count = 32
        dataset = []
        for row in rows:
            line = []
            for i in range(word_count):
                line.append(row[data_name_list[i]])
            dataset.append(line)
        return dataset

    def train(self, training_data: dict):
        for k, v in training_data.items():
            print(f'periodic cmd:{k}')
            print(f'label len:{len(v["labels"])}')
            print(f'Counter : {Counter(v["labels"])}')
            dataset = self.__prepross_data(k, v['characters'])
            self.df[k].fit(dataset, v['labels'])
            print(tree.export_text(self.df[k], show_weights=True))

    def detect_msg(self, last_msg, cur_msg):
        if cur_msg not in self.a_cmds:
            return False
        trigger_cmd = last_msg['CMD1']
        if trigger_cmd not in self.t_cmds:
            return False
        test_data = self.__prepross_data(trigger_cmd, [last_msg])
        result = self.df[trigger_cmd].predict(test_data)[0]
        return result == cur_msg
