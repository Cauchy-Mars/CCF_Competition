import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compress_24m(data):
    total_row = data.shape[0]
    unique_num = (int)(total_row / 24)
    range_row = range(unique_num)
    print(range_row)

    #确定列名
    new_column = ['province', 'adcode', 'model', 'bodyType']
    for i in range(1, 25):
        new_column.append('salesVolume' + str(i))
        new_column.append('popularity' + str(i))

    #进行拼接与压缩
    new_data = pd.DataFrame(columns=new_column)
    for i in range_row:
        temp = pd.DataFrame(columns=new_column)
        temp.loc[0, 'province'] = data.loc[24*i, 'province']
        temp.loc[0, 'adcode'] = data.loc[24*i, 'adcode']
        temp.loc[0, 'model'] = data.loc[24*i, 'model']
        temp.loc[0, 'bodyType'] = data.loc[24*i, 'bodyType']
        temp.loc[0, 'salesVolume1'] = data.loc[24*i, 'salesVolume']
        temp.loc[0, 'popularity1'] = data.loc[24*i, 'popularity']
        base = 24*i
        for j in range(1, 24):
            temp.loc[0, 'salesVolume' + str(j+1)] = data.loc[base+j, 'salesVolume']
            temp.loc[0, 'popularity' + str(j+1)] = data.loc[base+j, 'popularity']
        new_data = pd.concat([new_data, temp])
        #print(temp)

    #返回新文件
    return new_data