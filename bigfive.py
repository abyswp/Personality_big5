# -*- coding: utf-8 -*-

import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import random

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

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(all_essays).toarray()
y = dataset.iloc[:, 2:7].values

for d in range(0, len(y)):
    for i in range(0, len(y[0])):
        if y[d][i] == 'y':
            y[d][i] = 1
        else:
            y[d][i] = 0   


y_req = []
for i in range(0, len(y)):
    ones = []
    for j in range(0, 5):
        if y[i][j]:
            ones.append(j)

    if len(ones) != 0:        
        index = random.randrange(len(ones))
        y_req.append(ones[index])

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_req, test_size=1/24)


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(criterion='entropy')
classifier.fit(X_train, y_train)


import pickle
pickle.dump(classifier, open('model.sav', 'wb'))


pred = classifier.predict_proba(X_test)

y_pred = []
for i in range(0, len(pred)):
    max_p = 0
    ind = 0
    for j in range(0, len(pred[i])):
        if pred[i][j] > max_p:
            max_p = pred[i][j]
            ind = j
    y_pred.append(ind)

print(y_pred)
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)