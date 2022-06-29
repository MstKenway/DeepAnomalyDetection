import json

from stan_markov.state import State


class Rule:
    def __init__(self, cmd, time_cycles, is_period):
        self.state = State(cmd)
        self.time_cycles = time_cycles
        self.is_period = is_period

    def __repr__(self):
        return json.dumps(f'{self.state}\n{self.time_cycles}')

    def match(self, s: State, time_cycle=0) -> bool:
        if self.state != s:
            return False
        if self.is_period:
            if time_cycle == -1:
                return True
            for t in self.time_cycles:
                # if t - 100000 <= time_cycle <= t + 100000:
                if t - 40 <= time_cycle:
                    return True
        else:
            return True
        return False
