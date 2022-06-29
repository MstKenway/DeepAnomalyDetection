import json


class State:
    def __init__(self, cmd):
        cmd = int(cmd)
        address = cmd >> 11
        tr = (cmd >> 10) & 1
        sub_address = (cmd >> 5) & 0x1f
        word_count = cmd & 0x1f
        if tr == 1:
            # RT 2 BC
            self.src_add = address
            self.src_sub_add = sub_address
            self.dst_add = -1
            self.dst_sub_add = -1
        else:
            # BC 2 RT
            self.src_add = -1
            self.src_sub_add = -1
            self.dst_add = address
            self.dst_sub_add = sub_address
        # mode code
        if sub_address != 0 and sub_address != 0x1f:
            self.is_mode_code = False
        else:
            self.is_mode_code = True
        # word count
        if self.is_mode_code and word_count == 0:
            self.word_count = 0
        else:
            self.word_count = 32 if word_count == 0 else word_count

    def __hash__(self):
        return hash(f'{self.src_add}{self.src_sub_add}{self.dst_add}{self.dst_sub_add}'
                    f'{self.is_mode_code}{self.word_count}')

    def __eq__(self, other):
        return self.src_add == other.src_add \
               and self.src_sub_add == other.src_sub_add \
               and self.dst_add == other.dst_add \
               and self.dst_sub_add == other.dst_sub_add \
               and self.is_mode_code == other.is_mode_code \
               and self.word_count == other.word_count

    def __repr__(self):
        return json.dumps(self.__dict__, indent=4)
