import pandas as pd
import time
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB # naive Bayes
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import numpy as np
from numpy import array
from sklearn.linear_model import SGDClassifier # SVM
import pickle
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
import nltk
import string
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from .preprocessing import *
from . import load_pickle as load
from sklearn.externals import joblib
from .load_data import *

stopwords  = sw.words('english')
punct      = string.punctuation
lemmatizer = WordNetLemmatizer()
my_dict = {}

def init_classifier_failcode(data, program):
	global problem_input, failcode_df, control_number_input, family_input

	print(f'Classifying data for: {program} program.')

	problem_input = data['PROBLEMDESCRIPTION']
	control_number_input = data['CONTROLNUMBER']
	family_input = data['FAMILY']

	# Preprocessing
	to_classify = init_preprocessing(problem_input)

	# Predict
	result_json = predict(to_classify, program)

	return result_json

def predict(to_classify, program):
	# Load dictionaries
	failcode_dict = dict_failcodes.get(program)
	samples_dict = dict_samples_failcode.get(program)
	accuracies_dict = dict_accuracies_failcode.get(program)

	# clf = joblib.load("app/model_pickle.pkl")
	predicted = load.clf.predict(to_classify)

	# Creating lists with Failcode accuracy and samples
	failcode_codes = []
	failcode_acc = []
	failcode_samples = []
	for f in predicted:
		failcode_codes.append(failcode_dict.get(f.upper()))
		failcode_acc.append(accuracies_dict.get(f.upper()))	
		failcode_samples.append(samples_dict.get(f.upper()))

	# Transform the list to numpy array
	failcode_acc = np.array(failcode_acc)

	# Create a DataFrame from the classified data and the input (PROBLEM DESCRITPION)
	results_dataframe = pd.DataFrame({
		'CONTROLNUMBER': control_number_input,
		'FAMILY': family_input,
		'PROBLEMDESCRIPTION': problem_input, 
		'FAILCODE': failcode_codes,
		'FAILCODEDESCRIPTION': predicted, 
		'FAILCODEACCURACY': failcode_acc,
		'SAMPLESFAILCODE': failcode_samples
	})

	# Criação do JSON com os dados classificados
	json_data = results_dataframe.to_json(orient="index")

	return json_data