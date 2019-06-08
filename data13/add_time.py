# -*-coding:utf-8 -*-
import pandas as pd
df = pd.read_excel('x_96_by_13.xlsx', header=None)
week_list = []
week_df = pd.DataFrame()
for i in range(0, len(df)):
    week_list.append((i + 2) % 7)

for i in range(0, 7):
    temp_01_w = []
    for j in range(0, len(week_list)):
        if(week_list[j] == i):
            temp_01_w.append(1)
        else:
            temp_01_w.append(0)

    temp_ws = pd.Series(temp_01_w)
    df = pd.concat([df, temp_ws], axis=1, ignore_index=True)

df.to_excel('x_96_by_13_with_week.xlsx')
# print(df)
