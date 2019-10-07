import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

trainFile = "D:/CCF/train_user_reply_data.csv"
data = pd.read_csv(trainFile)
#print(data)

array = data.values

#print(array)
# newsReplyVolume = array[:,4]
# newsReplyVolume.dtype = 'int'
# print(newsReplyVolume)
# print(newsReplyVolume.dtype)

# model = array[:0]
# print(model)
# print(model.dtype)

carCommentVolume = array[:,3]
# print(carCommentVolume)
plt.scatter(carCommentVolume,carCommentVolume)

# plt.scatter(newsReplyVolume,newsReplyVolume)
plt.show()


