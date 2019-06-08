#coding=utf-8
import pandas as pd

SKIP_DAYS = 13

class DataProcess:
    def WeatherProcess(self, filename='data/STLF_DATA_IN_1.xls'):
        df = pd.read_excel(filename, sheet_name='气象数据', header=None, \
            names=['date', 'category', 'value'])
        df_group = df.groupby('date')
        df_weather = pd.DataFrame(columns=('date', 'avg_t','max_t','min_t', 'wet'))
        # print(df_group)
        index = 0
        for each_group in df_group:
            date = each_group[0]
            avg_t = each_group[1]['value'][3 * index]
            max_t = each_group[1]['value'][3 * index+1]
            min_t = each_group[1]['value'][3 * index+2]
            df_weather = df_weather.append(pd.DataFrame({'date':[date], 'avg_t':[avg_t], \
                'max_t':[max_t], 'min_t':[min_t]}), ignore_index=True)
            index += 1
        df_weather.to_excel('weather.xls')
    
    def FeatureForm(self, filename='data/STLF_DATA_IN_1.xls'):
        name = []
        for i in range(0, 96):
            name.append(str(i))
        df = pd.read_excel(filename, sheer_name='负荷数据', header = None, names=name)
        # print(df)
        df_y = df.iloc[SKIP_DAYS:-1, ]
        df_y.to_excel('data13/y_96.xlsx', index=False)
        # print(df_y)
        # input()

        df_x = pd.DataFrame()
        total_data_days = len(df)
        for i in range(SKIP_DAYS, total_data_days - 1):
            temp_df = pd.DataFrame()
            for j in range(0, SKIP_DAYS):
                temp_df = pd.concat([temp_df, df.iloc[i + (j - SKIP_DAYS), ]], axis=0, ignore_index=True, sort=False)
            df_x = pd.concat([df_x, temp_df.T], axis=0, ignore_index=True, sort=False)
            # print(df_x)
            # input()
        # print(df_x)
        df_x.to_excel('data13/x_96_by_13.xlsx', index=False)
        

if __name__ == '__main__':
    m_d = DataProcess()
    # m_d.WeatherProcess('STLF_DATA_IN_1.xls')
    m_d.FeatureForm()
