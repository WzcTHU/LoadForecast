#coding=utf-8
import pandas as pd

class DataCut():
    def __init__(self, filename_x, filename_y):
        self.train_xset = []
        self.validation_xset = []
        self.test_xset = []
        self.train_yset = []
        self.validation_yset = []
        self.test_yset = []
        self.df_x = pd.read_excel(filename_x)
        self.df_y = pd.read_excel(filename_y)
        self.total_len = len(self.df_y)

    def cut(self, train_len=0.99, validation_len=0.01, test_len=0):
        train_num = int(self.total_len * train_len)
        # validation_num = int(self.total_len * validation_len)
        validation_num = self.total_len - train_num
        test_num = self.total_len - train_num - validation_num
        for i in range(0, train_num):
            self.train_xset.append(list(self.df_x.ix[i]))
            self.train_yset.append(list(self.df_y.ix[i]))
        for i in range(train_num, train_num + validation_num):
            self.validation_xset.append(list(self.df_x.ix[i]))
            self.validation_yset.append(list(self.df_y.ix[i]))
        for i in range(train_num + validation_num, train_num + validation_num + test_num):
            self.test_xset.append(list(self.df_x.ix[i]))
            self.test_yset.append(list(self.df_y.ix[i]))
        return self.train_xset, self.train_yset, self.validation_xset, self.validation_yset

        
# if __name__ == '__main__':
#     test_cut = DataCut('x.csv', 'y.csv')
#     test_cut.cut()
#     print(len(test_cut.train_xset), len(test_cut.validation_xset), len(test_cut.test_xset))
