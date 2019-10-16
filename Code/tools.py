import numpy as np
import pandas as pd

def transform_type(data, columns, target_type='float'):
    '''将dataframe中的某些列的值的类型改变为target_type'''
    for c in columns:
        data[c] = data[c].astype(target_type)
    
    return data