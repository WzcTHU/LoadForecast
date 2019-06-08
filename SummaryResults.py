#coding=utf-8
import torch
import matplotlib.pyplot as plt
import numpy as np
import math

class SummaryResults():
    def __init__(self, label_y=[1], fore_y=[1]):
        self.accuracy = 0
        self.label_y = label_y
        self.fore_y = fore_y
        self.residual_list = []
        self.variance = 0
        self.label_y_combine = []
        self.fore_y_combine = []
        for each in self.label_y:
            self.label_y_combine.extend(each)
        for each in self.fore_y:
            self.fore_y_combine.extend(each)

    def cal_accuracy(self):
        n = len(self.label_y)
        daily_accuracy = []
        for i in range(n):
            sigma_d_2 = 0
            for j in range(0, 96):
                d = 100 * abs(self.label_y[i][j] - self.fore_y[i][j]) / abs(self.label_y[i][j])
                sigma_d_2 += pow(d, 2)
            daily_accuracy.append((100 - math.sqrt(sigma_d_2 / 96)))
        acc_sum = 0
        for each in daily_accuracy:
            acc_sum += each
        self.accuracy = (acc_sum / n)

    def get(self):
        self.cal_accuracy()
        print('The Accuracy is:', self.accuracy)
        fig1 = plt.figure()
        l1 = plt.plot(self.label_y_combine, marker='*', label='actual', lw=1, ms=3)
        l2 = plt.plot(self.fore_y_combine, marker='o', label='forecast', lw=1, ms=3)
        plt.legend()
        plt.show()
    
    def cal_residual(self):
        for i in range(0, len(self.label_y_combine)):
            self.residual_list.append(self.fore_y_combine[i] - self.label_y_combine[i])
        return self.residual_list

    def cal_variance(self):
        self.cal_residual()
        for i in range(0, len(self.residual_list)):
            self.variance += pow(self.residual_list[i], 2)
        return self.variance / len(self.residual_list)

