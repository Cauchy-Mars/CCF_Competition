import pandas as pd
import numpy as np
from encode import encode_train_test
from compress import compress_24m
from generate_train_from_24m import generate_train

#首先对训练集和测试集都进行编码
path = '../Data/'
train_data, eval_data = encode_train_test(path+'Handle/sales+search.csv', path+'evaluation_public.csv')

#将每个省，每种车型的数据压缩成一行数据
train_data = compress_24m(train_data)

#按照时间滑窗12个月生成训练集
train_data = generate_train(train_data)

#输出训练集
train_data.to_csv(path+'Final/train_sales_search_model_1.csv', index=False)