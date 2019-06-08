#coding=utf-8
from DataStandardScaler import *
from DataCut import *
from SummaryResults import *
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib
import scipy.io as sio
from sklearn.multioutput import MultiOutputRegressor
# 前33个特征固定不动，滚动预测的时候对后12个lmp值进行滚动（从前往后依次为前1小时到前12小时）
print('Cutting dataset...')
data = DataCut('data/x_with_week.xlsx', 'data/y_96.xlsx')
data.cut()

print('Data standardizating...')
data_scaler = DataStandardScaler(data.train_xset, data.train_yset, 
    data.validation_xset, data.validation_yset)

print('Linear Regression training...')
regressor = LinearRegression(n_jobs = -1)
regressor = MultiOutputRegressor(regressor)
regressor.fit(data_scaler.x_train_standard, data_scaler.y_train_standard)
joblib.dump(regressor, 'models/linear_model.m')

print('Forecasting...')
print(data_scaler.x_train_standard)
input()
print(data_scaler.x_validation_standard)
input()
y_fore_train = regressor.predict(data_scaler.x_train_standard)
y_fore_validation = regressor.predict(data_scaler.x_validation_standard)
data_scaler.reverse_trans(y_fore_train, y_fore_validation)


print('Getting results...')
sum_res_train = SummaryResults(data.train_yset, data_scaler.rev_y_train)
sum_res_validation = SummaryResults(data.validation_yset, data_scaler.rev_y_validation)
sio.savemat('ForecastResult/Validation/Linear.mat', {'Linearfore': data_scaler.rev_y_validation})
sum_res_train.get()
sum_res_validation.get()
res_list =  sum_res_validation.cal_residual()
sio.savemat('res/Linearres.mat', {'Linear_res': res_list})
print('The Variance is: ', sum_res_validation.cal_variance())
