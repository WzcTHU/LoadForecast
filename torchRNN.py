#coding=utf-8
import torch
import torch.utils.data as Data
import numpy as np
import torch.nn as nn
from DataStandardScaler import *
from DataCut import *
from SummaryResults import *
import scipy.io as sio


class TDataset(Data.Dataset):
    """Dataset wrapping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
    """

    def __init__(self, data_tensor, target_tensor):
        assert data_tensor.size(0) == target_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(input_size=1255, 
            hidden_size=2048, num_layers=2, batch_first=True, dropout=0.5)
        self.out_layer = nn.Linear(2048, 96)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out_layer(r_out[:, -1, :])
        return out
#----------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    print('Cutting dataset...')
    data = DataCut('data/x_96_by_7.xlsx', 'data/y_96.xlsx')
    data.cut()
    print('Data standardizating...')
    data_scaler = DataStandardScaler(data.train_xset, data.train_yset, 
        data.validation_xset, data.validation_yset)

    x_train_tensor = torch.from_numpy(np.array(data_scaler.x_train_standard)).float()
    y_train_tensor = torch.from_numpy(np.array(data_scaler.y_train_standard)).float()
    x_validation_tensor = torch.from_numpy(np.array(data_scaler.x_validation_standard)).float()
    y_validation_tensor = torch.from_numpy(np.array(data_scaler.y_validation_standard)).float()

    dataset = TDataset(data_tensor=x_train_tensor, target_tensor=y_train_tensor)

    BS = 64
    data_loader = Data.DataLoader(dataset=dataset, batch_size=BS, shuffle=True, drop_last=True)

    RNN = RNN()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LR = 0.001
    optimizer = torch.optim.Adam(RNN.parameters(), lr=LR)
    loss_func = nn.MSELoss()

    for epoch in range(300):
        # if epoch % 50 == 0:
        #     LR = LR * 0.5
        for step, (x, b_label) in enumerate(data_loader):
            b_x = x.view(BS, -1, len(x_train_tensor[0])).to(device)
            b_y = b_label.to(device)

            rnn_x = RNN(b_x)

            loss = loss_func(rnn_x, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % 100 == 0:
                print('Epoch:', epoch, '|train loss: %4f' % loss.data.numpy())

    torch.save(RNN.state_dict(), 'torch_models/LSTM_NET.pkl')
    #-------------------------------------------------------------------------------------------------------
    LSTM_net = RNN
    LSTM_net.load_state_dict(torch.load('torch_models/LSTM_NET.pkl'))
    # LSTM_net.eval()
    y_fore_validation = []
    y_fore_train = []

    for each in x_validation_tensor:
        each = each.view(1, -1, len(x_train_tensor[0]))
        y_fore_validation.append(LSTM_net(torch.from_numpy(np.array(each)).float()).detach().numpy())

    for each in x_train_tensor:
        each = each.view(1, -1, len(x_train_tensor[0]))
        y_fore_train.append(LSTM_net(torch.from_numpy(np.array(each)).float()).detach().numpy())

    data_scaler.reverse_trans(y_fore_train, y_fore_validation)
    y_fore_validation2 = []
    y_fore_train2 = []
    for each in data_scaler.rev_y_validation:
        y_fore_validation2.append(each[0])
    for each in data_scaler.rev_y_train:
        y_fore_train2.append(each[0])
    print('Getting results...')
    sum_res_train = SummaryResults(data.train_yset, y_fore_train2)
    sum_res_validation = SummaryResults(data.validation_yset, y_fore_validation2)

    sio.savemat('ForecastResult/Validation/RNN.mat', {'RNNfore': y_fore_validation2})
    sum_res_train.get()
    sum_res_validation.get()
    res_list =  sum_res_validation.cal_residual()
    sio.savemat('res/RNNres.mat', {'RNN_res': res_list})
    print(sum_res_validation.cal_variance())

