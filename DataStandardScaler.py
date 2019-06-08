#coding=utf-8
from sklearn.preprocessing import StandardScaler

class DataStandardScaler():
    def __init__(self, train_xset=[], train_yset=[], 
        validation_xset=[], validation_yset=[], 
        test_xset=[], test_yset=[]):
        self.scaler1 = StandardScaler().fit(train_xset)
        self.scaler2 = StandardScaler().fit(train_yset)
        self.x_train_standard = self.scaler1.transform(train_xset)
        self.y_train_standard = self.scaler2.transform(train_yset)
        self.x_validation_standard = self.scaler1.transform(validation_xset)
        self.y_validation_standard = self.scaler2.transform(validation_yset)
        # self.x_test_standard = self.scaler1.transform(test_xset)
        # self.y_test_standard = self.scaler2.transform(test_yset)
        # self.rev_x_train = []
        self.rev_y_train = []
        # self.rev_x_validation = []
        self.rev_y_validation = []
        # self.rev_x_test = []
        self.rev_y_test = []

    def reverse_trans(self, y_fore_train=[], y_fore_validation=[], y_fore_test=[], \
        valid_only=0, test_only=0):
        if(valid_only==0 & test_only==0):
            # self.rev_x_train = self.scaler1.inverse_transform(x_fore_train)
            self.rev_y_train = self.scaler2.inverse_transform(y_fore_train)
            # self.rev_x_validation = self.scaler1.inverse_transform(x_fore_validation)
            self.rev_y_validation = self.scaler2.inverse_transform(y_fore_validation)
            # self.rev_x_test = self.scaler1.inverse_transform(x_fore_test)
            # self.rev_y_test = self.scaler2.inverse_transform(y_fore_test)
            return self.rev_y_train, self.rev_y_validation
        if(valid_only==1):
            self.rev_y_validation = self.scaler2.inverse_transform(y_fore_validation)
            return self.rev_y_validation
        if(test_only==1):
            self.rev_y_test = self.scaler2.inverse_transform(y_fore_test)
            return self.rev_y_test

