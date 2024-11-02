import pandas as pd

df1 = pd.read_csv('data/taipei/attractions_0.csv')
df2 = pd.read_csv('data/taipei/attractions_1.csv')
df3 = pd.read_csv('data/taipei/attractions_2.csv')
df = pd.concat([df1, df2], ignore_index=True)
df = pd.concat([df, df3], ignore_index=True)

df.to_csv('data/taipei/attractions.csv', index=False, encoding='utf-8-sig')