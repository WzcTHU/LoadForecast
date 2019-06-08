#coding=utf-8
import pandas as pd
import re
import csv
SKIP_DAYS = 6
SKIP_HOURS = 24 * SKIP_DAYS

def add_6_hours(filename_x, filename_total_lmp):
    df_x = pd.read_csv(filename_x)
    df_total_lmp = pd.read_csv(filename_total_lmp)
    for i in range(1, 25):
        temp_df = df_total_lmp.iloc[SKIP_HOURS-i:-1*i, 1].reset_index()
        df_x = pd.concat([df_x, temp_df.iloc[:, 1]], axis=1)
    df_x.to_csv('data/x_add6.csv')

def peak_hour_from_data(filename):
    df = pd.read_csv(filename)['projected_peak_datetime_ept']
    time_list = []
    halfday_list = []
    for i in range(0, len(df)):
        time_list.append(int(re.findall(r'(\d+):00:00', df.ix[i])[0]))
        halfday_list.append(re.findall(r':00:00 (\w+)', df.ix[i])[0])
    peak_list = []
    for i in range(0, len(time_list)):
        if(halfday_list[i] == 'PM'):
            peak_list.append(time_list[i] + 12)
        else:
            peak_list.append(time_list[i])
    temp_zero_list = [0] * 24
    peak_list_bi = []
    for i in range(0, len(time_list)):
        temp_zero_list = [0] * 24
        temp_zero_list[peak_list[len(time_list)-i-1] - 1] = 1
        peak_list_bi += temp_zero_list

    with open('data/peak_bi.csv', 'w', newline='') as csvfile:
        writer  = csv.writer(csvfile)
        for row in peak_list_bi:
            writer.writerow([row])

if __name__ == '__main__':
    # add_6_hours('data/x.csv', 'data/total_lmp_data_20171001_20181101_processed.csv')
    peak_hour_from_data('data/projected_peak_raw.csv')
