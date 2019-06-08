#coding=utf-8
from DataStandardScaler import *
from DataCut import *
from SummaryResults import *
from sklearn import svm
from sklearn.externals import joblib
import scipy.io as sio
from sklearn.multioutput import MultiOutputRegressor

print('Cutting dataset...')
data = DataCut('data13/x_96_by_13_with_week.xlsx', 'data13/y_96.xlsx')
data.cut()

print('Data standardizating...')
data_scaler = DataStandardScaler(data.train_xset, data.train_yset, 
    data.validation_xset, data.validation_yset)

print('SVM training...')
# regressor = svm.SVR(C=0.1, gamma=0.01, max_iter=10000, kernel='rbf')
regressor = svm.SVR()
regressor = MultiOutputRegressor(regressor)
regressor.fit(data_scaler.x_train_standard, data_scaler.y_train_standard)

joblib.dump(regressor, 'models/svm_model.m')

print('Forecasting...')
y_fore_train = regressor.predict(data_scaler.x_train_standard)
y_fore_validation = regressor.predict(data_scaler.x_validation_standard)
data_scaler.reverse_trans(y_fore_train, y_fore_validation)
# data_scaler.reverse_trans(y_fore_validation=y_fore_validation, valid_only=1)

print('Getting results...')
sum_res_train = SummaryResults(data.train_yset, data_scaler.rev_y_train)
sum_res_validation = SummaryResults(data.validation_yset, data_scaler.rev_y_validation)
sio.savemat('ForecastResult/Validation/SVM.mat', {'SVMfore': data_scaler.rev_y_validation})
sum_res_train.get()
sum_res_validation.get()
res_list =  sum_res_validation.cal_residual()
sio.savemat('res/SVMres.mat', {'SVM_res': res_list})
print('The Variance is: ', sum_res_validation.cal_variance())