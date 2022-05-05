from torch import nn
from torch.utils.data import Dataset


class RNN(nn.Module):
    def __init__(self, input_size,algo='gru'):
        super(RNN, self).__init__()
        if algo=='gru':
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=128,
                num_layers=1,
                batch_first=True,
            )
        elif algo=='rnn':
            self.rnn = nn.RNN(
                input_size=input_size,
                hidden_size=128,
                num_layers=1,
                batch_first=True,
            )
        elif algo=='lstm':
             self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=128,
                num_layers=1,
                batch_first=True,
            )
        self.out = nn.Sequential(
            nn.Linear(128, 4),
        )
        self.hidden = None

    def forward(self, x):
        r_out, self.hidden = self.rnn(x)  # None 表示 hidden state 会用全0的 state
        out = self.out(r_out)
        return out


class TrainSet(Dataset):
    def __init__(self, data, lables):
        # 定义好 image 的路径
        self.data, self.label = data.float(), lables.float()

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

