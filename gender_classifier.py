from flask import Flask
from flask.ext.cors import CORS 
import pandas as pd
import numpy as np
import flask
import requests

from nltk.tokenize import RegexpTokenizer

from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)
CORS(app)
url = 'http://0.0.0.0:5000/'

males = pd.read_csv('Indian-Male-Names.csv')
females = pd.read_csv('Indian-Female-Names.csv')

males.drop(['race'], axis=1, inplace=True)
females.drop(['race'], axis=1, inplace=True)

males['gender'] = males['gender'].map({'m':0})
females['gender'] = females['gender'].map({'f':1})

males = males.dropna()
females = females.dropna()

tokenizer = RegexpTokenizer(r'\w+')

name = []
for i in females['name']:
    l = tokenizer.tokenize(i)
    if len(l[0]) < 4:
        if len(l) == 1:
            l = l[0]
        else:
            l = l[1]
    else:
        l = l[0]
    name.append(l)

females['name'] = name
females = females.dropna()

name = []
for i in males['name']:
    l = tokenizer.tokenize(i)
    if len(l[0]) < 4:
        if len(l) == 1:
            l = l[0]
        else:
            l = l[1]
    else:
        l = l[0]
    name.append(l)

males['name'] = name
males = males.dropna()

data = pd.concat([males, females], axis=0)
data = data.iloc[np.random.permutation(len(data))]
data.index = np.arange(0,data.shape[0])

bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\w', min_df=1)

train_x = np.c_[np.array(bigram_vectorizer.fit_transform(data['name']).toarray())]
train_y = np.array(data['gender'])

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=5.0)

clf.fit(train_x, train_y)

#l = []
#l.append(input("Enter first name"))
#print(l)
#print(clf.predict(np.c_[np.array(bigram_vectorizer.transform(l).toarray())]))

def gender(l):
	return clf.predict(np.c_[np.array(bigram_vectorizer.transform(l).toarray())])


@app.route('/<name>')
def reccomend(name):
	l = []
	l.append(name)
	pred = gender(l)
	print(pred)
	if pred == 0:
		return 'Male'
	elif pred == 1:
		return 'Female'
	else:
		return 'invalid input! please pass first name e.g: /shivam'


@app.route('/')
def test():
	return '"/first_name" return "gender". Eg: "/Priya" returns "Female" | "/Raju" returns "Male"'

if __name__=="__main__":
	app.run(host='0.0.0.0',debug=True)