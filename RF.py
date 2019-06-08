#coding=utf-8
from DataStandardScaler import *
from DataCut import *
from SummaryResults import *
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.externals import joblib
import scipy.io as sio

print('Cutting dataset...')
data = DataCut('data13/x_96_by_13_with_week.xlsx', 'data13/y_96.xlsx')
data.cut()

print('Data standardizating...')
data_scaler = DataStandardScaler(data.train_xset, data.train_yset, 
    data.validation_xset, data.validation_yset)

#训练时启用start
print('RF training...')
regressor = RandomForestRegressor(n_estimators=64, n_jobs=-1)
# regressor = RandomForestRegressor()
regressor.fit(data_scaler.x_train_standard, data_scaler.y_train_standard)
joblib.dump(regressor, 'models/rf_model.m')
# 滚动预测并滚动修改后续特征向量，滚动周期为24
print('RF forecasting train set...')
y_fore_train = regressor.predict(data_scaler.x_train_standard)
#训练时启用end
#---------------------------------------------------------------------
# y_fore_validation = np.array([])
# for i in range(0, len(data_scaler.x_validation_standard)):
#     y_fore_validation = np.append(y_fore_validation, regressor.predict(data_scaler.x_validation_standard[i].reshape(1, -1)))
#     # print(y_fore_validation)
#     # input()
#     print(data_scaler.x_validation_standard[i+1])
#     input()
#     print(i)
#     if((i+1)%24 != 0):
#         for j in range(0, i%24+1):
#             if(i<len(data_scaler.x_validation_standard)-1):
#                 data_scaler.x_validation_standard[i+1][33+i%24-j] = y_fore_validation[i-i%24+j]
#     print(data_scaler.x_validation_standard[i+1])
#     input()
#---------------------------------------------------------------------
print('RF forecasting validation set...')
#调用现有模型start
# m_rf = joblib.load('rf_model_s.m')
# y_fore_train = m_rf.predict(data_scaler.x_train_standard)
# y_fore_validation = m_rf.predict(data_scaler.x_validation_standard)
#调用现有模型end

y_fore_validation = regressor.predict(data_scaler.x_validation_standard)
data_scaler.reverse_trans(y_fore_train, y_fore_validation)

print('Getting results...')
sum_res_train = SummaryResults(data.train_yset, data_scaler.rev_y_train)
sum_res_validation = SummaryResults(data.validation_yset, data_scaler.rev_y_validation)
sio.savemat('ForecastResult/Validation/RF.mat', {'RFfore': data_scaler.rev_y_validation})
sum_res_train.get()
sum_res_validation.get()
res_list =  sum_res_validation.cal_residual()
sio.savemat('res/RFres.mat', {'RF_res': res_list})
print('The Variance is: ', sum_res_validation.cal_variance())
