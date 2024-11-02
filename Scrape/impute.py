import pandas as pd
import numpy as np
import chardet
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder
from gensim.models import Word2Vec

with open('classified_data.csv', 'rb') as f:
    result = chardet.detect(f.read())

encoding = result['encoding']
df = pd.read_csv('classified_data.csv', encoding=encoding)
# word to vector
sentences = [name.split() for name in df['basicName']]
model_w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

def get_average_vector(words, model):
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if len(word_vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(word_vectors, axis=0)

df['basicName_vector'] = df['basicName'].apply(lambda x: get_average_vector(x.split(), model_w2v))
# print(df['basicName_vector'])
# 展開 basicName_vector 為多個單獨的特徵
vector_size = df['basicName_vector'].iloc[0].shape[0]  # 確認向量的維度
# for i in range(vector_size):
#     df[f'basicName_vector_{i}'] = df['basicName_vector'].apply(lambda x: x[i])
vector_df = pd.DataFrame(df['basicName_vector'].tolist(), columns=[f'basicName_vector_{i}' for i in range(vector_size)])
df = pd.concat([df, vector_df], axis=1)
# 移除原始的 basicName_vector
df = df.drop(columns=['basicName_vector'])

categories = ['活動', '觀光', '餐飲', '購物']
encoder = OneHotEncoder(handle_unknown='ignore', categories=[categories], sparse_output=False).set_output(transform='pandas')
encoded_labels = encoder.fit_transform(df[['label']])
df = pd.concat([df, encoded_labels], axis=1)

days = ['mon', 'tues', 'wed', 'thurs', 'fri', 'sat', 'sun']
features_to_impute = ['rating', 'recommend'] + [f'{day}{suffix}' for day in days for suffix in ['Open', 'Close']]
label_encoded = ['label_活動', 'label_觀光', 'label_餐飲', 'label_購物']
X = df[[f'basicName_vector_{i}' for i in range(vector_size)] + label_encoded + features_to_impute].copy()
X_masked = X.copy()
X_masked[features_to_impute] = X_masked[features_to_impute].replace(0, np.nan)
# print(X[features_to_impute + label_encoded])
# exit()
imputer = IterativeImputer(max_iter=10, random_state=0)
imputed_data = imputer.fit_transform(X_masked)

imputed_df = pd.DataFrame(imputed_data, columns=[f'basicName_vector_{i}' for i in range(vector_size)] + label_encoded + features_to_impute)
imputed_df[features_to_impute] = np.where(X[features_to_impute] == 0, 0, imputed_df[features_to_impute])
# imputed_df['basicName_vector'] = imputed_df['basicName_vector'].apply(lambda x: np.array(x))

df.update(imputed_df)

def minutes_to_open_time(minutes):
    if minutes is None or pd.isna(minutes):
        return None
    if isinstance(minutes, str):
        minutes = int(minutes)
    hours = minutes // 60
    mins = minutes % 60
    if mins >= 30:
        mins = 30
    else:
        mins = 0
    return f'{int(hours):02d}:{int(mins):02d}'

def minutes_to_close_time(minutes):
    if minutes is None or pd.isna(minutes): 
        return None
    if isinstance(minutes, str):
        minutes = int(minutes)
    hours = minutes // 60
    mins = minutes % 60
    if mins >= 30:
        mins = 30
    else:
        mins = 0
    return f'-{int(hours):02d}:{int(mins):02d}'

for day in ['mon', 'tues', 'wed', 'thurs', 'fri', 'sat', 'sun']:
    df[day + 'Time'] = df[day + 'Open'].apply(minutes_to_open_time)
    df[day + 'Time'] += df[day + 'Close'].apply(minutes_to_close_time)
    df[day + 'Time'] = df.apply(
        lambda row: 0 if row[day + 'Open'] == 0 and row[day + 'Close'] == 0 else row[day + 'Time'], axis = 1
    )

def convert_to_hours(time_str):
    result = float(time_str)
    result += 0.25
    result *= 10
    temp = result % 5
    result = int(result - temp)
    result = float(result) / 10
    if result < 0.5:
        result = 0.5
    if result > 10:
        result = 10
    
    return result

df['recommend'] = df['recommend'].apply(convert_to_hours)
df['rating'] = df['rating'].astype(float).round(1)
df = df.drop(columns=[f'basicName_vector_{i}' for i in range(vector_size)] + label_encoded)
drop_df = (df[['monTime', 'tuesTime', 'wedTime', 'thursTime', 'friTime', 'satTime', 'sunTime']] == '0').all(axis=1)
df = df[~drop_df]
df.reset_index(drop=True, inplace=True)

df.to_csv('imputed_data.csv', index=False, encoding='utf-8-sig')