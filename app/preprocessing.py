import pandas as pd
import numpy as np
from numpy import array
import nltk
import string
from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag
import string

stopwords = sw.words('english')
stopwords_pt = set(sw.words('portuguese'))

punct = string.punctuation
lemmatizer = WordNetLemmatizer()
my_dict = {}

def init_preprocessing(list):
	df = create_dataframe(list)
	result = preprocessing(df)
	return result

def create_dataframe(list):
	df = pd.DataFrame(list)
	df.columns = ['PROBLEMDESCRIPTION']
	return df

def preprocessing(df):
	read_create_df()
	create_dict()
	preprocessing_result = do_preprocessing(df)
	return preprocessing_result

def inverse_transform(X):
	return [" ".join(doc) for doc in X]

def transform(X):
	return [
		list(tokenize(doc)) for doc in X
	]

def replace_words(word):
	if my_dict.get(word):
		str = my_dict.get(word)
	else:
		str = word
	return str

def lemmatize(token, tag):
	tag = {
		'N': wn.NOUN,
		'V': wn.VERB,
		'R': wn.ADV,
		'J': wn.ADJ
	}.get(tag[0], wn.NOUN)
	return WordNetLemmatizer().lemmatize(token, tag)

def tokenize(document):
	# Break the document into sentences
	for sent in nltk.sent_tokenize(document):
		# Break the sentence into part of speech tagged tokens
		for token, tag in nltk.pos_tag(nltk.wordpunct_tokenize(sent)):
			# Remove spaces before and after in string value
			token = token.strip()
			token=token.lower()
			
			token = replace_words(token)

			if token in stopwords:
				continue

			if token in stopwords_pt:
				continue	

			if all(char in punct for char in token):
				continue

			lemma = lemmatize(token, tag)
			yield lemma

def read_create_df():
	global xls #
	xls = pd.ExcelFile('app/data/processed/dictionary.xls')
	global dictionary 
	dictionary = pd.read_excel(xls)


def create_dict():
	for f, t in zip(dictionary['ERRADO'], dictionary['CERTO']):
		f = str(f)
		t = str(t)
		f = f.strip()
		t = t.strip()
		my_dict[f] = t


def do_preprocessing(df_test):
	global preprocessing_test
	preprocessing_test = df_test['PROBLEMDESCRIPTION']

	#print(preprocessing_test)
	preprocessing_test = transform(preprocessing_test)
	preprocessing_test = inverse_transform(preprocessing_test)
	preprocessing_test = array(preprocessing_test)
	preprocessing_test = pd.Series(preprocessing_test)
	return preprocessing_test
	#print(my_dict)
	#print(type(preprocessing_test))
	# print(preprocessing_test.to_json(orient="index"))
	# return preprocessing_test.to_json(orient="index")

