import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
import nltk
import string
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from numpy import array
import numpy as np
import os.path
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.naive_bayes import ComplementNB

stopwords  = set(sw.words('english'))
stopwords_pt =set(sw.words('portuguese'))
stopwords_sp =set(sw.words('spanish'))

punct      = set(string.punctuation)
lemmatizer = WordNetLemmatizer()


filename_sample_sheets = {
    'Phenom': 'phenom_samples.xlsx',
    'Legacy 500': 'legacy500_samples.xlsx',
    'Legacy 600': 'legacy600_samples.xlsx',
    'Lineage': 'lineage_samples.xlsx'
}

filename_accuracy_sheets = {
    'Phenom': 'phenom_accuracies.xlsx',
    'Legacy 500': 'legacy500_accuracies.xlsx',
    'Legacy 600': 'legacy600_accuracies.xlsx',
    'Lineage': 'lineage_accuracies.xlsx'
}

paths_pickles = {
    'Phenom': '../../../../models/finalfix/phenom_finalfix_clf.pkl',
    'Legacy 500': '../../../../models/finalfix/legacy500_finalfix_clf.pkl',
    'Legacy 600': '../../../../models/finalfix/legacy600_finalfix_clf.pkl',
    'Lineage': '../../../../models/finalfix/lineage_finalfix_clf.pkl'
}

def create_dict():
    for f, t in zip(dictionary['ERRADO'], dictionary['CERTO']):
        my_dict[f] = t

def replace_words(word):
    if my_dict.get(word):
        str = my_dict.get(word)
    else:
        str = word
    return str

def inverse_transform(X):
    return [" ".join(doc) for doc in X]

def transform(X):
    return [
        list(tokenize(doc)) for doc in X
    ]

def listarDocumentos(document):
    return document

def tokenize(document):
    # Break the document into sentences
    for sent in nltk.sent_tokenize(document):
        # Break the sentence into part of speech tagged tokens
        for token, tag in nltk.pos_tag(nltk.wordpunct_tokenize(sent)):
            # Remove spaces before and after in string value
            token = token.strip()
            token = token.lower()

            token = replace_words(token)

            if token in stopwords:
                continue

            if token in stopwords_pt:
                continue

            if token in stopwords_sp:
                continue

            if all(char in punct for char in token):
                continue

            lemma = lemmatize(token, tag)
            yield lemma

def lemmatize(token, tag):
    tag = {
        'N': wn.NOUN,
        'V': wn.VERB,
        'R': wn.ADV,
        'J': wn.ADJ
    }.get(tag[0], wn.NOUN)
    return WordNetLemmatizer().lemmatize(token, tag)

def do_preprocessing(df):
#    data_preprocessing = df['PROBLEM DESCRIPTION']
    data_preprocessing = transform(df)
    data_preprocessing = inverse_transform(data_preprocessing)
    data_preprocessing = array(data_preprocessing)
    data_preprocessing = pd.Series(data_preprocessing)
    return data_preprocessing

def main():
    global dictionary, my_dict, df_samples, predicted_data, df_test
    global dict_accuracies, dict_accuracies_excel, df_accuracy

    # Create the dictionary
    xls_dict = pd.ExcelFile('../dictionary.xls')
    dictionary = pd.read_excel(xls_dict)
    my_dict = {}
    create_dict()

    # Read the Dataset
    xls_dataset = pd.ExcelFile('dataset.xlsx')

    for read in ['Phenom','Legacy 500','Legacy 600' ,'Lineage']:

        df_training = pd.read_excel(xls_dataset,read)
        df_training = df_training.reindex(np.random.permutation(df_training.index)) # Shuffle the DF

        # Limit the PROBLEM DESCRIPTION length to 400 characters
        df_training['SUBJECT/CORRECTIVE ACTION'] = df_training['SUBJECT/CORRECTIVE ACTION'].str.slice(0,400).head(len(df_training))

        # Eliminates leading and trailing spaces from each value of the FAILCODE DESCR column
        df_training['FINAL FIX DESCRIPTION'] = df_training['FINAL FIX DESCRIPTION'].str.strip()

        # Creating cross validation method
        x_val = df_training
        # Split the dataset into Training and Test data
        train_length = int(0.8 * len(df_training)) # Train data is 80% of the dataset
        df_training_cutted = df_training[0 : train_length]
        df_test = df_training[train_length : ]

        # Get the filename of the sheets
        program = df_training['PROGRAM'][0]
        filename_samples = filename_sample_sheets.get(program)
        filename_accuracies = filename_accuracy_sheets.get(program)
        
        # Create pickle path
        path_pickle = paths_pickles.get(program)
        path_pickle = os.path.join(os.path.dirname(__file__), path_pickle)

        # Create the sheet with the sample values
        df_samples = df_training['FINAL FIX DESCRIPTION'].value_counts().to_frame()
        df_samples = df_samples.reset_index()
        df_samples.columns = ['FINAL FIX DESCRIPTION', 'SAMPLES']

        # Create samples Excel file
        writer = pd.ExcelWriter(filename_samples, engine='xlsxwriter')
        df_samples.to_excel(writer, sheet_name='Samples')
        writer.save()
        #print(df_training_cutted['PROBLEM DESCRIPTION'])
        # Apply preprocessing
        df_training_cutted_processed = do_preprocessing(df_training_cutted['SUBJECT/CORRECTIVE ACTION'])

        # Apply preprocessing in cross validation
        x_val_processed = do_preprocessing(x_val['SUBJECT/CORRECTIVE ACTION'])

        pl = { 'Phenom' :[
            ('vect', CountVectorizer(ngram_range=(1,2))),
            ('tfidf',TfidfTransformer()),
            ('clf',OneVsRestClassifier(LinearSVC(class_weight='balanced', max_iter=10000), n_jobs=-1))] ,
            'Legacy 500' :[
            ('vect', CountVectorizer(ngram_range=(1,2))),
            ('tfidf',TfidfTransformer()),
            ('clf',OneVsRestClassifier(LinearSVC(class_weight='balanced', max_iter=10000), n_jobs=-1))] ,
            'Legacy 600' :[
            ('vect', CountVectorizer(ngram_range=(1,2))),
            ('tfidf',TfidfTransformer()),
            ('clf',OneVsRestClassifier(LinearSVC(class_weight='balanced', max_iter=10000), n_jobs=-1))] ,
            'Lineage' :[
            ('vect', CountVectorizer(ngram_range=(1,2))),
            ('tfidf',TfidfTransformer()),
            ('clf',OneVsRestClassifier(LinearSVC(class_weight='balanced', max_iter=10000), n_jobs=-1))] 
        }


        # Model pipeline ComplementNB(norm=True, alpha=0.8))
        text_clf = Pipeline(pl[read])

        # Calculate the accuracy with k-fold cross validation method
        k_fold = KFold(n_splits=5)
        score = cross_val_score(text_clf,x_val_processed, x_val['FINAL FIX DESCRIPTION'] , cv = k_fold)
        
        print(f'{program} x-val accuracy {np.mean(score)} (+/-) {score.std()*2}')

        # Fit the model
        text_clf.fit(df_training_cutted_processed, df_training_cutted['FINAL FIX DESCRIPTION'])

        # Predict the test data
        predicted_data = text_clf.predict(df_test['SUBJECT/CORRECTIVE ACTION'])

        # Calculate the accuracy
        accuracy = np.mean (predicted_data == df_test['FINAL FIX DESCRIPTION'])*100

        print(f'{read} Model accuracy: {accuracy}')

        dict_accuracies = {}
        for predicted, correct in zip(predicted_data, df_test['FINAL FIX DESCRIPTION']):
            #print(f'{predicted}, {correct}')

            if not predicted in dict_accuracies:
                dict_accuracies[predicted] = {}
                dict_accuracies[predicted]['correct'] = 0
                dict_accuracies[predicted]['total'] = 0

            if predicted == correct:
                dict_accuracies[predicted]['correct'] += 1

            dict_accuracies[predicted]['total'] += 1

        dict_accuracies_excel = {}
        for failcode in dict_accuracies:
            correct = dict_accuracies[failcode].get('correct')
            total = dict_accuracies[failcode].get('total')
            accuracy = round((correct/total)*100, 2)
            dict_accuracies_excel[failcode] = f'{accuracy}%'

        
        df_accuracy = pd.DataFrame.from_dict(dict_accuracies_excel, orient='index')
        df_accuracy = df_accuracy.reset_index()
        df_accuracy.columns = ['FINAL FIX DESCRIPTION', 'ACCURACY']

        # Create accuracy Excel file    
        writer = pd.ExcelWriter(filename_accuracies, engine='xlsxwriter')
        df_accuracy.to_excel(writer, sheet_name='Accuracies')
        writer.save()

        ## Pickle
        joblib.dump(text_clf, path_pickle, compress=True)

if __name__ == "__main__":
    main()