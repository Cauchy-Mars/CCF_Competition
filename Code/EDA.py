import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from BaseModel import BaseModel
#读入三份训练数据
model = BaseModel()
model.genDataset()
print(model.train.head())