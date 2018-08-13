# 统计召回率
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, accuracy_score, f1_score

df = pd.read_csv('test_out_epoch_132.csv')
print(len(df))

df = df.drop_duplicates(['file']) # 去重后
print(len(df))
# print(len(df[df['groundtruth'] == df['predict']]))

# y_true = np.zeros([len(df)])

# for i in range(len(df)):
    # y_true[i] = df['groundtruth'][i][1]
#
y_true = df['groundtruth']
y_pred = df['predict']

recall = recall_score(y_true, y_pred, average='macro')
acc = accuracy_score(y_true, y_pred)
f1_score = f1_score(y_true, y_pred, average='weighted')

print('  recall: ', recall)
print(' accuacy: ', acc)
print('f1_score: ', f1_score)
