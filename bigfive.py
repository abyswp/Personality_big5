# -*- coding: utf-8 -*-

import pandas as pd
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle

dataset = pd.read_csv('essays.csv', encoding='cp1252')
indices = []

for i in range(0, len(dataset)):
    if dataset['cEXT'][i] == 'n' and dataset['cNEU'][i] == 'n' and dataset['cAGR'][i] == 'n'and dataset['cCON'][i] == 'n' and dataset['cOPN'][i] == 'n':
        indices.append(i)

dataset.drop(dataset.index[indices], inplace=True)
dataset = dataset.reset_index(drop=True)

all_essays = []

for i in range(0, len(dataset['TEXT'])):
    essay = re.sub('a-zA-Z', ' ', dataset['TEXT'][i])
    essay = essay.lower()
    essay = essay.split()
    ps = PorterStemmer()
    wnl = WordNetLemmatizer()

    essay = [wnl.lemmatize(word) if wnl.lemmatize(word).endswith('e') else ps.stem(word) for word in essay if not word in set(stopwords.words())]
    essay = ' '.join(essay)
    all_essays.append(essay)

    print("Done " + str(i))

with open("essaysfinal", "wb") as fp:
    pickle.dump(all_essays, fp)

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

classifier = RandomForestClassifier(criterion='entropy')
classifier.fit(X, y_req)

pickle.dump(classifier, open('model.sav', 'wb'))