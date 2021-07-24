import torch
from torch import nn

class full_conn(nn.Module): # Inspired from Fastai's fully connected module
    def __init__(self, hyper_params): # hyper_params (dict): {'in', 'H', 'out', 'drop_ps'} (drop_ps is list of size 2)
        super(full_conn, self).__init__()
        self.bn1 = nn.BatchNorm1d(hyper_params['in'], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.lin1 = nn.Linear(hyper_params['in'], hyper_params['H'][0], bias=False)
        self.drop1 = nn.Dropout(p=hyper_params['drop_p'][0])
        self.relu = nn.ReLU()

        self.bn2 = nn.BatchNorm1d(hyper_params['H'][0], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.lin2 = nn.Linear(hyper_params['H'][0], hyper_params['out'], bias=False)
        self.drop2 = nn.Dropout(p=hyper_params['drop_p'][1])

    def forward(self, x):
        x = self.bn1(x)
        x = self.drop1(self.lin1(x))
        x = self.relu(x)

        x = self.bn2(x)
        x = self.drop2(self.lin2(x))
        return x
    
class BiLSTM(nn.Module):
    def __init__(self, hyper_params): # hyper_params (dict): {'in', 'H', 'out', 'LSTM_num_layers', 'drop_ps'} (drop_ps is list of size 1)
        super(BiLSTM, self).__init__()
        self.input_size = hyper_params['in']
        self.hidden_size = hyper_params['H'][0]
        self.num_LSTM_layers = hyper_params['LSTM_num_layers']

        self.bn1 = nn.BatchNorm1d(hyper_params['in'], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.LSTM = nn.LSTM(hyper_params['in'], 
                            hyper_params['H'][0], 
                            num_layers = hyper_params['LSTM_num_layers'], 
                            batch_first=True,
                            dropout=hyper_params['drop_ps'][0], 
                            bidirectional=True)
        self.lin = nn.Linear(hyper_params['H'][0], hyper_params['out'])

    def forward(self, x):
        x = self.bn1(x).reshape(-1, self.input_size, self.input_size) # Seq Len = Input Size

        h0 = torch.zeros(2 * self.num_LSTM_layers, x.shape[0], self.hidden_size)
        c0 = torch.zeros(2 * self.num_LSTM_layers, x.shape[0], self.hidden_size)
        out, _ = self.LSTM(x, (h0, c0))

        out = self.lin(out[:, -1, :]) # Last timestep
        return out

