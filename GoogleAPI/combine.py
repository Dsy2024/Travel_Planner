import pandas as pd
import chardet

with open('model_data/taipei/attraction.csv', 'rb') as f:
    result = chardet.detect(f.read())

encoding = result['encoding']
df = pd.read_csv('model_data/taipei/attraction.csv', encoding=encoding)

df.to_csv('model_data/taipei/attraction.csv', index=False, encoding='utf-8-sig')