#coding=utf-8
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from DataStandardScaler import *
from DataCut import *
from SummaryResults import *
import scipy.io as sio
from torchRNN import RNN
import torch

W = [0.0000, 0.1820, 0.0246, 0.4424, 0.0000, 0.3509]
# LGBM, MLP, RF, SVM, XGB, RNN
# W = sio.loadmat('res/models_W.mat').get('W_value')
print('loading models...')
LGBM = joblib.load('models/lgbm_model.m')
MLP = joblib.load('models/mlp_model.m')
RF = joblib.load('models/rf_model.m')
SVM = joblib.load('models/svm_model.m')
XGB = joblib.load('models/xgb_model.m')
RNN = RNN()
LSTM_net = RNN
LSTM_net.load_state_dict(torch.load('torch_models/LSTM_NET.pkl'))
print('Cutting dataset...')
data = DataCut('data13/x_96_by_13_with_week.xlsx', 'data13/y_96.xlsx')
data.cut()

print('Data standardizating...')
data_scaler = DataStandardScaler(data.train_xset, data.train_yset, 
    data.validation_xset, data.validation_yset)

scaler_x = data_scaler.scaler1
scaler_y = data_scaler.scaler2

print('reading input x...')
df = pd.read_excel('final_x.xlsx', header=None)
m_x = []
for i in range(0, 13):
    for j in range(0, 96):
        # print(df[i][j])
        # input()
        m_x.append(df[j][i])

m_x.extend([0,0,0,1,0,0,0])
x_standard = scaler_x.transform([m_x])
# print(x_standard)
y_LGBM = LGBM.predict(x_standard)
y_MLP = MLP.predict(x_standard)
y_RF = RF.predict(x_standard)
y_SVM = SVM.predict(x_standard)
y_XGB = XGB.predict(x_standard)
y_RNN = []


x_RNN = torch.from_numpy(np.array(x_standard)).float()
for each in x_RNN:
    each = each.view(1, -1, len(x_RNN[0]))
    y_RNN.append(LSTM_net(torch.from_numpy(np.array(each)).float()).detach().numpy())

y_combine = []

for i in range(0, len(y_LGBM)):
    y_combine.append(W[0] * y_LGBM[i] + W[1] * y_MLP[i]+ W[2] * y_RF[i] + \
        W[3] * y_SVM[i] + W[4] * y_XGB[i] + W[5] * y_RNN[i])

y = scaler_y.inverse_transform(y_combine)
print(y)
sy = pd.Series(y[0])
sy.to_csv('final_y.csv', index=False)
# print('Getting results...')
# sum_res_validation = SummaryResults(data.validation_yset, y_combine)
# print(sum_res_validation.cal_variance())
# sum_res_validation.get()