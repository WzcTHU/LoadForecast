#coding=utf-8
from lightgbm import LGBMRegressor
from DataStandardScaler import *
from DataCut import *
from SummaryResults import *
from sklearn.externals import joblib
import scipy.io as sio
from sklearn.multioutput import MultiOutputRegressor
# 前33个特征固定不动，滚动预测的时候对后12个lmp值进行滚动（从前往后依次为前1小时到前12小时）
print('Cutting dataset...')
data = DataCut('data13/x_96_by_13_with_week.xlsx', 'data13/y_96.xlsx')
data.cut()

print('Data standardizating...')
data_scaler = DataStandardScaler(data.train_xset, data.train_yset, 
    data.validation_xset, data.validation_yset)

print('LGBM training...')
# regressor = LGBMRegressor(n_estimators=64)
regressor = LGBMRegressor()
regressor = MultiOutputRegressor(regressor)
regressor.fit(data_scaler.x_train_standard, data_scaler.y_train_standard)
joblib.dump(regressor, 'models/lgbm_model.m')

print('Forecasting...')
y_fore_train = regressor.predict(data_scaler.x_train_standard)
y_fore_validation = regressor.predict(data_scaler.x_validation_standard)
data_scaler.reverse_trans(y_fore_train, y_fore_validation)

print('Getting results...')
sum_res_train = SummaryResults(data.train_yset, data_scaler.rev_y_train)
sum_res_validation = SummaryResults(data.validation_yset, data_scaler.rev_y_validation)
sio.savemat('ForecastResult/Validation/LGBM.mat', {'LGBMfore': data_scaler.rev_y_validation})
sum_res_train.get()
sum_res_validation.get()
res_list =  sum_res_validation.cal_residual()
sio.savemat('res/LGBMres.mat', {'LGBM_res': res_list})
print('The Variance is: ', sum_res_validation.cal_variance())