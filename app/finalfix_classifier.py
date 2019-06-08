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
from pandas.io.json import json_normalize

stopwords  = sw.words('english')
punct      = string.punctuation
lemmatizer = WordNetLemmatizer()
my_dict = {}


def separate_dataset(data, string):
	#separate dataset by corrective action equals to string
	dataset = pd.DataFrame({
		'CONTROLNUMBER': data['CONTROLNUMBER'],
		'SUBJECT': data['SUBJECT'],
		'CORRECTIVEACTION': data['CORRECTIVEACTION'],
		'FAMILY': data['FAMILY']
	})

	dataset1 = dataset.loc[dataset['CORRECTIVEACTION'] != string]

	dataset2 = dataset.loc[dataset['CORRECTIVEACTION'] == string]

	return dataset1, dataset2


def fill_finalfix(dataset, finalfix):
	#create descriptions for finalfix
	return [finalfix for x in range(len(dataset['CONTROLNUMBER']))]



def init_classifier_finalfix(data, program):
	global problem_input, problem_df, finalfix_df, control_number_input, family_input, subject, c_action, dataset1, dataset2
	
	#dataset1, dataset2 = separate_dataset(data, '')
	
	dataset = pd.DataFrame({
		'CONTROLNUMBER': data['CONTROLNUMBER'],
		'SUBJECT': data['SUBJECT'],
		'CORRECTIVEACTION': data['CORRECTIVEACTION'],
		'FAMILY': data['FAMILY']
	})

	print(f'Classifying data for: {program} program.')

	# maintaining row order
	subject = dataset['SUBJECT']
	#subject = subject.append(dataset2['SUBJECT'], ignore_index=True) 
	# maintaining row order
	c_action = dataset['CORRECTIVEACTION']
	#c_action = c_action.append(dataset2['CORRECTIVEACTION'], ignore_index=True) 
	# maintaining row order
	control_number_input = dataset['CONTROLNUMBER']
	#control_number_input = control_number_input.append(dataset2['CONTROLNUMBER'], ignore_index=True) 
	# maintaining row order
	family_input = dataset['FAMILY'] 
	#family_input = family_input.append(dataset2['FAMILY'], ignore_index=True) 
	# concating two columns
	problem_input = []
	for sub, action in zip(dataset['SUBJECT'], dataset['CORRECTIVEACTION']):
		problem_input.append(str(sub) + " / " + str(action))
	# Preprocessing
	to_classify = init_preprocessing(problem_input)
	
	# Predict
	result_json = predict(to_classify, program)

	return result_json


def predict(to_classify, program):
	# Load dictionaries
	finalfix_dict = dict_finalfix.get(program)
	samples_dict = dict_samples_finalfix.get(program)
	accuracies_dict = dict_accuracies_finalfix.get(program)

	# clf = joblib.load("app/model_pickle.pkl")
	predicted = load.clf.predict(to_classify)

	#concat descriptions
	predict_ff = pd.Series(predicted)
	#ff_description = predict_ff.append(pd.Series(fill_finalfix(dataset2, 'UNDER INVESTIGATION')), ignore_index=True)
	ff_description = predict_ff

	# Creating lists with Final fix code accuracy and samples
	finalfix_codes = []
	finalfix_acc = []
	finalfix_samples = []

	for f in ff_description:
		finalfix_codes.append(finalfix_dict.get(f.upper()))
		finalfix_acc.append(accuracies_dict.get(f.upper()))	
		finalfix_samples.append(samples_dict.get(f.upper()))
	# Transform the list to numpy array
	finalfix_acc = np.array(finalfix_acc)
	
	# Create a DataFrame from the classified data and the input (PROBLEM DESCRITPION)
	results_dataframe = pd.DataFrame({
		'CONTROLNUMBER': control_number_input,
		'FAMILY': family_input,
		'SUBJECT': subject,
		'CORRECTIVEACTION': c_action,  
		'FINALFIXCODE': finalfix_codes,
		'FINALFIXDESCRIPTION': ff_description, 
		'FINALFIXACCURACY': finalfix_acc,
		'SAMPLESFINALFIX': finalfix_samples
	})


	# Criação do JSON com os dados classificados
	json_data = results_dataframe.to_json(orient="index")

	return json_data



# handle_load_pickle('Phenom', 'finalfix')
# # Data to classify
# to_classify = pd.read_excel('testePH300.xlsx')  
# # Classify:
# result = init_classifier_finalfix(to_classify, 'Phenom')

# df = pd.read_json(result, orient='index', )

# writer = pd.ExcelWriter('teste.xlsx')
# df.to_excel(writer, index=False)
# writer.save()