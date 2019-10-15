import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def encode_train_test(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    province = LabelEncoder()
    adcode = LabelEncoder()
    model = LabelEncoder()
    bodyType = LabelEncoder()
    province.fit(train['province'])
    adcode.fit(train['adcode'])
    model.fit(train['model'])
    bodyType.fit(train['bodyType'])
    
    train['province'] = province.transform(train['province'])
    train['adcode'] = adcode.transform(train['adcode'])
    train['model'] = model.transform(train['model'])
    train['bodyType'] = bodyType.transform(train['bodyType'])
    
    test['province'] = province.transform(test['province'])
    test['adcode'] = adcode.transform(test['adcode'])
    test['model'] = model.transform(test['model'])
    
    return train, test