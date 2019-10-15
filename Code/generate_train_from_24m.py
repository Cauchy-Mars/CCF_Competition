import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from BaseModel import BaseModel

def concat_x_y(data, train_columns=[], label_column=[], final_columns=[]):
    if (len(final_columns) != (len(train_columns) + len(label_column))):
        print('Uncompatible length for final_columns!')
    new_data = pd.concat([data[train_columns], data[label_column]], axis=1)
    new_data.columns = final_columns
    return new_data

def generate_train(data):
    #读入数据
    base_columns = ['province', 'adcode', 'model', 'bodyType']
    fix1 = 'salesVolume'
    fix2 = 'popularity'
    change_columns = []
    for i in range(1, 13):
        change_columns.append(fix1 + str(i))
        change_columns.append(fix2 + str(i))
    final_columns = base_columns+change_columns+['pred']
    data_1 = concat_x_y(data, base_columns+change_columns, [fix1+str(13)], final_columns)
    change_columns = []
    for i in range(2, 14):
        change_columns.append(fix1 + str(i))
        change_columns.append(fix2 + str(i))
    data_2 = concat_x_y(data, base_columns+change_columns, [fix1+str(14)], final_columns)
    change_columns = []
    for i in range(3, 15):
        change_columns.append(fix1 + str(i))
        change_columns.append(fix2 + str(i))
    data_3 = concat_x_y(data, base_columns+change_columns, [fix1+str(15)], final_columns)
    change_columns = []
    for i in range(4, 16):
        change_columns.append(fix1 + str(i))
        change_columns.append(fix2 + str(i))
    data_4 = concat_x_y(data, base_columns+change_columns, [fix1+str(16)], final_columns)
    change_columns = []
    for i in range(5, 17):
        change_columns.append(fix1 + str(i))
        change_columns.append(fix2 + str(i))
    data_5 = concat_x_y(data, base_columns+change_columns, [fix1+str(17)], final_columns)
    change_columns = []
    for i in range(6, 18):
        change_columns.append(fix1 + str(i))
        change_columns.append(fix2 + str(i))
    data_6 = concat_x_y(data, base_columns+change_columns, [fix1+str(18)], final_columns)
    change_columns = []
    for i in range(7, 19):
        change_columns.append(fix1 + str(i))
        change_columns.append(fix2 + str(i))
    data_7 = concat_x_y(data, base_columns+change_columns, [fix1+str(19)], final_columns)
    change_columns = []
    for i in range(8, 20):
        change_columns.append(fix1 + str(i))
        change_columns.append(fix2 + str(i))
    data_8 = concat_x_y(data, base_columns+change_columns, [fix1+str(20)], final_columns)
    change_columns = []
    for i in range(9, 21):
        change_columns.append(fix1 + str(i))
        change_columns.append(fix2 + str(i))
    data_9 = concat_x_y(data, base_columns+change_columns, [fix1+str(21)], final_columns)
    change_columns = []
    for i in range(10, 22):
        change_columns.append(fix1 + str(i))
        change_columns.append(fix2 + str(i))
    data_10 = concat_x_y(data, base_columns+change_columns, [fix1+str(22)], final_columns)
    change_columns = []
    for i in range(11, 23):
        change_columns.append(fix1 + str(i))
        change_columns.append(fix2 + str(i))
    data_11 = concat_x_y(data, base_columns+change_columns, [fix1+str(23)], final_columns)
    change_columns = []
    for i in range(12, 24):
        change_columns.append(fix1 + str(i))
        change_columns.append(fix2 + str(i))
    data_12 = concat_x_y(data, base_columns+change_columns, [fix1+str(24)], final_columns)

    #按行拼接所有数据
    final = pd.concat([data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, data_10, data_11, data_12], axis=0)
    
    return final