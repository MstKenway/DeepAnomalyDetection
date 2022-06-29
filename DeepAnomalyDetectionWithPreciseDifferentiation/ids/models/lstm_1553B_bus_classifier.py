from torch import nn


# LSTM模型
class BusClassifierLSTMNet(nn.Module):
    bus_classifier_name = 'LSTM'

    def __init__(self, input_size, hidden_size, output_size=2, batch_first=True, num_layers=2,
                 bidirectional=False, dropout=0.5):
        super(BusClassifierLSTMNet, self).__init__()
        self.input_size = input_size  # 输入的特征的个数
        self.hidden_size = hidden_size  # 隐藏层的维度
        self.n_directions = 2 if bidirectional else 1
        self.num_layers = num_layers  # 有多少层RNN
        self.output_size = output_size
        self.batch_first = batch_first

        # 定义LSTM
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            batch_first=self.batch_first, bidirectional=bidirectional, dropout=dropout)
        # 最后一层线性层，输入为hidden_size * self.n_directions,，输出为output_size消息标签
        self.fc = nn.Linear(self.hidden_size * self.n_directions, self.output_size)
        self.relu = nn.ReLU()

    def forward(self, input, hidden=None):
        gru_out, _ = self.lstm(input, None)
        fc_output = self.fc(self.relu(gru_out[:, -1, :]))  # 最后来个全连接层,确保层想要的维度（类别数）
        return fc_output
