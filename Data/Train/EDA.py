import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#读入三份训练数据
sale = pd.read_csv('Train/train_sales_data.csv')
search = pd.read_csv('Train/train_search_data.csv')
reply = pd.read_csv('Train/train_user_reply_data.csv')
print('sale_data shape: ' + str(sale.shape))
print('search_data shape: ' + str(search.shape))
print('user_reply_data shape: ' + str(reply.shape))


