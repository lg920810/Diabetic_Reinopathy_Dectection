
import pandas as pd
import os
'''训练集验证集划分8:2'''
# csv_path = '~/lg/DR_datasets_1024/trainLabels.csv'
# data = pd.read_csv(csv_path)
# train = data.sample(frac=0.8)
# result = pd.merge(data, train, on='image', how='left')
# test = result[result.isnull().values==True].drop(['level_y'], axis=1)
# test.rename(columns={'level_x':'level'}, inplace=True)
# print(test.columns)
# train.to_csv('csv/train.csv', index=False)
# test.to_csv('csv/test.csv', index=False)
#
'''训练集类平衡，每类10000'''
# csv_path = 'csv/train.csv'
# df = pd.read_csv( csv_path)
#
# df_0 = df[df['level'] == 0]
# df_1 = df[df['level'] == 1]
# df_2 = df[df['level'] == 2]
# df_3 = df[df['level'] == 3]
# df_4 = df[df['level'] == 4]
# df_0 = df_0.sample(n=10000)
# # print()
# frame_1 = [df_1,df_1,df_1,df_1,df_1,df_1]
# result_1 = pd.concat(frame_1, axis=0).sample(n=10000)
#
# frame_2 = [df_2,df_2,df_2]
# result_2 = pd.concat(frame_2, axis=0).sample(n=10000)
#
# frame_3 = [df_3,df_3,df_3,df_3,df_3,df_3,df_3,df_3,df_3,df_3,df_3,df_3,df_3,df_3,df_3]
# result_3 = pd.concat(frame_3, axis=0).sample(n=10000)
#
# frame_4 = [df_4,df_4,df_4,df_4,df_4,df_4,df_4,df_4,df_4,df_4,df_4,df_4,df_4,df_4,df_4,df_4,df_4,df_4]
# result_4 = pd.concat(frame_4, axis=0).sample(n=10000)
#
# frame = [df_0, result_1, result_2, result_3, result_4]
# result = pd.concat(frame, axis=0).sample(frac=1)
# result.to_csv('csv/train_balance_50000.csv')

'''验证集平衡，每类140'''
# csv_path = 'csv/test.csv'
# df = pd.read_csv( csv_path)
#
# df_0 = df[df['level'] == 0]
# df_1 = df[df['level'] == 1]
# df_2 = df[df['level'] == 2]
# df_3 = df[df['level'] == 3]
# df_4 = df[df['level'] == 4]
#
# df_0 = df_0.sample(n=140)
# df_1 = df_1.sample(n=140)
# df_2 = df_2.sample(n=140)
# df_3 = df_3.sample(n=140)
# df_4 = df_4.sample(n=140)
#
# frame = [df_0, df_1, df_2, df_3, df_4]
# result = pd.concat(frame, axis=0).sample(frac=1)
# result.to_csv('csv/test_balance_700.csv')



csv_path = 'csv/train_gan.csv'
df = pd.read_csv(csv_path)

df_0 = df[df['level'] == 0]
df_1 = df[df['level'] == 1]
df_2 = df[df['level'] == 2]
df_3 = df[df['level'] == 3]
df_4 = df[df['level'] == 4]
df_0 = df_0.sample(n=10000)
# print()
frame_1 = [df_1,df_1,df_1,df_1,df_1,df_1]
result_1 = pd.concat(frame_1, axis=0).sample(n=10000)

frame_2 = [df_2,df_2,df_2]
result_2 = pd.concat(frame_2, axis=0).sample(n=10000)

frame_3 = [df_3,df_3,df_3,df_3,df_3,df_3,df_3,df_3,df_3,df_3,df_3,df_3,df_3,df_3,df_3]
result_3 = pd.concat(frame_3, axis=0).sample(n=10000)

frame_4 = [df_4,df_4,df_4,df_4,df_4,df_4,df_4,df_4,df_4,df_4,df_4,df_4,df_4,df_4,df_4,df_4,df_4,df_4]
result_4 = pd.concat(frame_4, axis=0).sample(n=10000)

frame = [df_0, result_1, result_2, result_3, result_4]
result = pd.concat(frame, axis=0).sample(frac=1)
result.to_csv('csv/train_balance_50000.csv')


