import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump
from joblib import load

# df = pd.read_csv('data/taipei/attractions_0.csv')
df = pd.read_csv('arranged_data.csv')

# word to vector
# sentences = [name.split() for name in df['basicName']]
# model_w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# def get_average_vector(words, model):
#     word_vectors = [model.wv[word] for word in words if word in model.wv]
#     if len(word_vectors) == 0:
#         return np.zeros(model.vector_size)
#     return np.mean(word_vectors, axis=0)

# df['basicName_vector'] = df['basicName'].apply(lambda x: get_average_vector(x.split(), model_w2v))
model = load('random_forest_model.joblib')

# print(df)

# classifier
X = df['basicName']
# X = list(df['basicName_vector'])
# y = df['label']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

vectorizer = TfidfVectorizer(max_features=200)
X_vec = vectorizer.fit_transform(X)
# X_test = vectorizer.transform(X_test)
y = model.predict(X_vec)
df['label'] = y

df.to_csv('classified_data.csv', index=False, encoding='utf-8-sig')
exit()

# 训练模型
model = RandomForestClassifier()
model.fit(X_vec, y)
dump(model, 'random_forest_model.joblib')

# 预测并评估
# y_pred = model.predict(X_test)
# print(classification_report(y_test, y_pred, zero_division=0))
# print('Accuracy:', accuracy_score(y_test, y_pred))

# model_mnb = MultinomialNB()
# model_mnb.fit(X_train, y_train)
# y_pred_mnb = model_mnb.predict(X_test)

# model_cnb = ComplementNB()
# model_cnb.fit(X_train, y_train)
# y_pred_cnb = model_cnb.predict(X_test)

# print('Multinomial Naive Bayes:')
# print(classification_report(y_test, y_pred_mnb))
# print('Accuracy:', accuracy_score(y_test, y_pred_mnb))

# print('Complement Naive Bayes:')
# print(classification_report(y_test, y_pred_cnb))
# print('Accuracy:', accuracy_score(y_test, y_pred_cnb))

# df.to_csv('filled_data.csv', index=False, encoding='utf-8-sig')