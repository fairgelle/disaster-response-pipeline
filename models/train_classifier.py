import sys
import nltk 
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
import pickle
import re

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.multioutput import MultiOutputClassifier
from sqlalchemy import create_engine, inspect
from utils import tokenize


def load_data(database_filepath):
    '''
    Args:
      database_filepath (str): Filepath of the database. Format: 'xxx.db'
    
    Returns:
      X (series): A series containing the predictor variable.
      Y (df): A dataframe containing one or more target variables.
      target_names (index): An index containing the column name of the target variables      
        
    '''
    # load data from database
    engine = create_engine('sqlite:///{filepath}'.format(filepath=database_filepath))
    df = pd.read_sql_table(database_filepath, engine)
    
    # defining the features (message) and the multi outcome variables 
    X = df.message
    
    # excluding columns that are not used for the Y variable
    cols_to_exclude = ['id', 'message', 'original', 'genre']
    Y = df.drop(cols_to_exclude, axis=1)

    Y['related'] = Y['related'].replace(2,1)
    target_names = Y.columns

    #target names are the labels of the multi outcome variables
    return X, Y, target_names
    

def build_model():
    '''
    A function to build a machine learning model used to predict 
    the label of disaster response messages. The model is using the following components:
      1. CountVectorizer: Create a document-term matrix where each observation reflects the count of
         tokens available in the sentence.
      2. TfidfTransformer: Normalizing document-term matrix using tf-idf to take into account
         overall document weightage of a word
      3. SGDClassifier: Stochastic Gradient Descent classifier used to predict the target variable given
         a tf-idf normalized document-term matrix
    '''
    
    pipeline = Pipeline([
      ('vect', CountVectorizer(tokenizer=tokenize)),
      ('tfidf', TfidfTransformer()),
      ('clf', MultiOutputClassifier(SGDClassifier()))
    ])
    
    parameters = {
      'vect__min_df': [0.5, 1],
      'vect__ngram_range': [(1, 1), (1, 2)],
      'clf__estimator__tol': [0.001, 0.01]
    }
    
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters, verbose=5)
    
    return cv

def evaluate_model(model, X_test, Y_test, target_names):
    '''
    Evaluating the model that is build by the build_model() function
    by printing out the precision, recall and f1-score values
    
    Params:
      model: The model that is returned by the build_model_function
      X_test: The portion of the predictor variable that is used to test the model
      Y_test: The portion of the target variable that is compared against the Y_pred generated
        when the model is used on the X_test
      target_names: the index containing the column names of the target variables that is returned
        by the load_data function        
    '''
    # predict on test data
    Y_pred = model.predict(X_test)
    
    for i in range(len(target_names)):
      print("Target Name: {}".format(target_names[i]))
      print("Classification Report: \n\n {}".format(
        classification_report(Y_test[target_names[i]], Y_pred.transpose()[i])
      ))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()