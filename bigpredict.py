# -*- coding: utf-8 -*-

import pickle
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

dataset = pd.read_csv('essays.csv', encoding='cp1252')
indices = []

for i in range(0, len(dataset)):
    if dataset['cEXT'][i] == 'n' and dataset['cNEU'][i] == 'n' and dataset['cAGR'][i] == 'n'and dataset['cCON'][i] == 'n' and dataset['cOPN'][i] == 'n':
        indices.append(i)

dataset.drop(dataset.index[indices], inplace=True)
dataset = dataset.reset_index(drop=True)

clss = pickle.load(open('model.sav', 'rb'), encoding='latin1')
all_essays = pickle.load(open('essaysfinal', 'rb'))

complete_ds = []
y_req = []

y = dataset.iloc[:, 2:7].values
for d in range(0, len(y)):
    for i in range(0, len(y[0])):
        if y[d][i] == 'y':
            y[d][i] = 1
        else:
            y[d][i] = 0

for i in range(0, len(y)):
    for j in range(0, len(y[i])):
        if y[i][j] == 1:
            complete_ds.append(all_essays[i])
            y_req.append(j)


cv = CountVectorizer()
X = cv.fit_transform(complete_ds).toarray()

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_req, test_size=1/24)

probs = clss.predict_proba(X_test)

pred = clss.predict(X_test)

corr = 0
wrng = 0
for i in range(0, len(pred)):
    if pred[i] == y_test[i]:
        corr += 1
    else:
        wrng += 1

print("Correct: " + str(corr))
print("Wrong: " + str(wrng))



###################################################
essay = ""
with open("profile.txt") as f:
    essay = f.read()

essay = re.sub('a-zA-Z', ' ', essay)
essay = essay.lower()
essay = essay.split()
ps = PorterStemmer()
wnl = WordNetLemmatizer()

essay = [wnl.lemmatize(word) if wnl.lemmatize(word).endswith('e') else ps.stem(word) for word in essay if not word in set(stopwords.words())]
essay = ' '.join(essay)

X = cv.transform([essay]).toarray()

predd = clss.predict(X)
preddd = clss.predict_proba(X)