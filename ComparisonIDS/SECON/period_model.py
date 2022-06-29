class P_Detect:
    def __init__(self):
        self.d = {}  # consist
        # detection var
        self.last = None
        self.detect_time_history = {}
        self.p_cmds = set()

    def clear(self):
        self.last = None
        self.detect_time_history = {}

    def detect_msg(self, cmd, time_stamp, threshold=1.0):
        if self.last is None:
            self.last = cmd
            self.detect_time_history[cmd] = time_stamp
            return True
        # 是否在链表上
        if cmd not in self.d[self.last]['next']:
            return False
        if self.detect_time_history.get(cmd) is not None:
            period_time = time_stamp - self.detect_time_history[cmd]
            # 是否周期长度合理
            if period_time < self.d[cmd]['min_time'] * threshold:
                return False
        # 合理则更新时间和上一条
        self.last = cmd
        self.detect_time_history[cmd] = time_stamp
        return True

    def train(self, training_data, period_cmds):
        self.p_cmds = set(period_cmds)
        time_history = {}
        last = None
        for row in training_data:
            cmd = row['CMD1']
            if last is None:
                last = cmd
                time_history[cmd] = row['TimeStamp']
                self.d[cmd] = {'next': set(), 'min_time': -1}
            else:
                if cmd not in period_cmds:
                    continue
                if time_history.get(cmd) is None:

                    self.d[cmd] = {'next': set(), 'min_time': -1}
                else:
                    period_time = row['TimeStamp'] - time_history[cmd]
                    if self.d[cmd]['min_time'] == -1 or self.d[cmd]['min_time'] > period_time:
                        self.d[cmd]['min_time'] = period_time
                self.d[last]['next'].add(cmd)
                last = cmd
                time_history[cmd] = row['TimeStamp']
