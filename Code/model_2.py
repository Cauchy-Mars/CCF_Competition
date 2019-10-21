import sys
import numpy as np
import pandas as pd
import os 
import gc
from tqdm import tqdm, tqdm_notebook
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import LabelEncoder
import datetime
import time
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

path  = '../Data/Train/'
train_sales  = pd.read_csv(path+'train_sales_data.csv')
train_search = pd.read_csv(path+'train_search_data.csv')
train_user   = pd.read_csv(path+'train_user_reply_data.csv')
evaluation_public = pd.read_csv(path+'evaluation_public.csv')
submit_example    = pd.read_csv(path+'submit_example.csv')
data = pd.concat([train_sales, evaluation_public], ignore_index=True)
data = data.merge(train_search, 'left', on=['province', 'adcode', 'model', 'regYear', 'regMonth'])
data = data.merge(train_user, 'left', on=['model', 'regYear', 'regMonth'])
data['label'] = data['salesVolume']
data['id'] = data['id'].fillna(0).astype(int)
#因为train_sales和evaluation_public上下连接在了一起，但是evaluation_public没有bodyType这一个特征，所以需要我们根据model进行补全
data['bodyType'] = data['model'].map(train_sales.drop_duplicates('model').set_index('model')['bodyType'])
#LabelEncoder
for i in ['bodyType', 'model']:
    data[i] = data[i].map(dict(zip(data[i].unique(), range(data[i].nunique()))))
data['mt'] = (data['regYear'] - 2016) * 12 + data['regMonth']

def get_stat_feature(df_): 
    '''每一个省的每一种车型在每一个月都有一条数据，该函数构造了从该月开始向后偏移1,2，……，12个月的销售量和流行程度。方法很巧妙'''
    df = df_.copy()
    stat_feat = []
    #构造出了两个组合特征model_adcode和model_adcode_mt
    df['model_adcode'] = df['adcode'] + df['model']
    df['model_adcode_mt'] = df['model_adcode'] * 100 + df['mt']
    for col in tqdm(['label','popularity']):
        # shift
        for i in [1,2,3,4,5,6,7,8,9,10,11,12]:
            stat_feat.append('shift_model_adcode_mt_{}_{}'.format(col,i))
            #此处最后+i就是为了最后构造偏移做准备，也就是绝对月份相同的同一个省的一种车型的model_adcode_mt_{}_{}一定是相同的
            df['model_adcode_mt_{}_{}'.format(col,i)] = df['model_adcode_mt'] + i
            df_last = df[~df[col].isnull()].set_index('model_adcode_mt_{}_{}'.format(col,i))
            df['shift_model_adcode_mt_{}_{}'.format(col,i)] = df['model_adcode_mt'].map(df_last[col])    
    return df,stat_feat

def score(data, pred='pred_label', label='label', group='model'):
    data['pred_label'] = data['pred_label'].apply(lambda x: 0 if x < 0 else x).round().astype(int)
    data_agg = data.groupby('model').agg({
        pred:  list,
        label: [list, 'mean']
    }).reset_index()
    data_agg.columns = ['_'.join(col).strip() for col in data_agg.columns]
    nrmse_score = []
    for raw in data_agg[['{0}_list'.format(pred), '{0}_list'.format(label), '{0}_mean'.format(label)]].values:
        nrmse_score.append(
            mse(raw[0], raw[1]) ** 0.5 / raw[2]
        )
    print(1 - np.mean(nrmse_score))
    return 1 - np.mean(nrmse_score)

def get_model_type(train_x,train_y,valid_x,valid_y,m_type='lgb'):   
    if m_type == 'lgb':
        model = lgb.LGBMRegressor(
                                num_leaves=2**5-1, reg_alpha=0.25, reg_lambda=0.25, objective='mse',
                                max_depth=-1, learning_rate=0.05, min_child_samples=5, random_state=2019,
                                n_estimators=2000, subsample=0.9, colsample_bytree=0.7,
                                )
        model.fit(train_x, train_y, 
              eval_set=[(train_x, train_y),(valid_x, valid_y)], 
              categorical_feature=cate_feat, 
              early_stopping_rounds=100, verbose=100)      
    elif m_type == 'xgb':
        model = xgb.XGBRegressor(
                                max_depth=5 , learning_rate=0.05, n_estimators=2000, 
                                objective='reg:gamma', tree_method = 'hist',subsample=0.9, 
                                colsample_bytree=0.7, min_child_samples=5,eval_metric = 'rmse' 
                                )
        model.fit(train_x, train_y, 
              eval_set=[(train_x, train_y),(valid_x, valid_y)], 
              early_stopping_rounds=100, verbose=100)   
    return model

def get_train_model(df_, m, m_type='lgb'):
    df = df_.copy()
    # 数据集划分
    st = 1
    all_idx   = (df['mt'].between(st , m-13))
    train_idx = (df['mt'].between(st , m-13))
    valid_idx = (df['mt'].between(m-12, m-12))
    test_idx  = (df['mt'].between(m  , m  ))
    print('all_idx  :',st ,m-1)
    print('train_idx:',st ,m-5)
    print('valid_idx:',m-4,m-4)
    print('test_idx :',m  ,m  )  
    # 最终确认
    train_x = df[train_idx][features]
    train_y = df[train_idx]['label']
    valid_x = df[valid_idx][features]
    valid_y = df[valid_idx]['label']   
    # get model
    model = get_model_type(train_x,train_y,valid_x,valid_y,m_type)  
    # offline
    df['pred_label'] = model.predict(df[features])
    best_score = score(df[valid_idx]) 
    # online
    if m_type == 'lgb':
        model.n_estimators = model.best_iteration_ + 100
        model.fit(df[all_idx][features], df[all_idx]['label'], categorical_feature=cate_feat)
    elif m_type == 'xgb':
        model.n_estimators = model.best_iteration + 100
        model.fit(df[all_idx][features], df[all_idx]['label'])
    df['forecastVolum'] = model.predict(df[features]) 
    print('valid mean:',df[valid_idx]['pred_label'].mean())
    print('true  mean:',df[valid_idx]['label'].mean())
    print('test  mean:',df[test_idx]['forecastVolum'].mean())
    # 阶段结果
    sub = df[test_idx][['id']]
    sub['forecastVolum'] = df[test_idx]['forecastVolum'].apply(lambda x: 0 if x < 0 else x).round().astype(int)  
    return sub,df[valid_idx]['pred_label']

def trend_factor(data):
    '''计算趋势因子'''
    for col in ['adcode', 'model', 'model_adcode']:
        temp_df = pd.DataFrame(columns=[col, 'factor_{}'.format(col)])
        year_1 = (data['mt'].between(1, 12))
        year_2 = (data['mt'].between(13, 24))
        i = 0
        for df in data[col].unique():
            temp1 = data[(data[col] == df) & (year_1)]
            temp2 = data[(data[col] == df) & (year_2)]
            sum1 = temp1['label'].sum()
            sum2 = temp2['label'].sum()
            factor = sum2 / sum1
            temp_df.loc[i] = {col:df, 'factor_{}'.format(col):factor}
            i = i+1
        data = data.merge(temp_df, how='left', on=[col])
    return data


for month in [25,26,27,28]: 
    m_type = 'xgb' 
    
    data_df, stat_feat = get_stat_feature(data)
    
    num_feat = ['regYear'] + stat_feat
    cate_feat = ['adcode','bodyType','model','regMonth']
    
    if m_type == 'lgb':
        for i in cate_feat:
            data_df[i] = data_df[i].astype('category')
    elif m_type == 'xgb':
        lbl = LabelEncoder()  
        for i in tqdm(cate_feat):
            data_df[i] = lbl.fit_transform(data_df[i].astype(str))
           
    features = num_feat + cate_feat
    print(len(features), len(set(features)))   
    
    sub,val_pred = get_train_model(data_df, month, m_type)
    #将预测出来的结果再重新加入训练文件，以得到下一个月的结果
    data.loc[(data.regMonth==(month-24))&(data.regYear==2018), 'salesVolume'] = sub['forecastVolum'].values
    data.loc[(data.regMonth==(month-24))&(data.regYear==2018), 'label'      ] = sub['forecastVolum'].values
ratio = trend_factor(data_df)
print('ratio is: ' + str(ratio))
sub = data.loc[(data.regMonth>=1)&(data.regYear==2018), ['id','salesVolume']]
sub.columns = ['id','forecastVolum']
sub['forecastVolum'].apply(lambda x: x * ratio)
sub[['id','forecastVolum']].round().astype(int).to_csv('../Data/Final/model_2_1.csv', index=False)

