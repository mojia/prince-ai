import pandas as pd
import numpy as np


df = pd.DataFrame({
    'name': ['wang', 'xin', 'kai', 'sheng', 'hai', 'heng', 'yue', 'guo', 'ping', 'en', 'dong', 'feng', 'qun'],
    'age': [18, 31, 27, None, 46, 14, None, 29, 54, None, 59, 32, 70],
    'sex': ['nan', 'nan', 'nue', 'nan', 'nue', 'nan', 'nue', 'nan', 'nue', 'nan', 'nue', 'nue', 'nue']
})

df['age_qcut'] = pd.qcut(df['age'], 5)
# print(df['age_qcut'])

df['age_cut'] = pd.cut(df['age'], 2)
# print(df['age_cut'])

age_avg = df['age'].mean()
age_std = df['age'].std()
age_none_count = df['age'].isnull().sum()
print('age_avg ' + str(age_avg))
print('age_std ' + str(age_std))
print('age_none_count ' + str(age_none_count))

age_none_new_list = np.random.randint(10, 10 + 4, size=10)
# print(age_none_new_list)

df['sex'] = df['sex'].map({'nan': 0, 'nue': 1}).astype(int)
# print(df['sex'])


print(df.shape)
