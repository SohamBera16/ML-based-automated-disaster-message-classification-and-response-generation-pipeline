# import libraries
import nltk 
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger') 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd 
pd.set_option('display.max_columns', 70)
import os
import sys
import re

from sqlalchemy import create_engine
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer

def load_data(database_filepath):
    """
    INPUT:
    file path for the database 
    
    OUTPUT:
    X - messages (input variable) 
    Y - categories of the messages (output variable)
    category_names - category name for Y
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_messages_table', engine)
    
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    """
    function to extract tokens from the input text
    
    Input:
        text: Text to be tokenized
    Output:
        clean_tokens: List of tokens extracted
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    class for extracting the starting verb 
    
    This class creating a new feature for the ML classifier by extracting the starting verb
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
    
def build_model():
    """
    INPUT:
    None
    
    OUTPUT:
    cv_new = ML model pipeline after performing grid search
    """
    pipeline_adaboost = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('starting_verb_transformer', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    parameters_new = {'clf__estimator__learning_rate': [0.3, 0.5],
                  'clf__estimator__n_estimators': [10, 20]}

    cv_new = GridSearchCV(pipeline_adaboost, param_grid = parameters_new) 
    
    return cv_new


def evaluate_model(model, X_test, y_test, category_names):
    """function for evaluating different metrics of the model performance
    
    inputs:
    model
    X_test
    y_test
    category_names
       
    Returns:
    data_metrics: accuracy, precision, recall and f1 scores.
    """

    y_pred = model.predict(X_test)
    
    metrics = []
    
    # Evaluate metrics for each set of labels
    for i, col in enumerate(category_names):
        accuracy = accuracy_score(y_test[col], y_pred[:, i])
        precision = precision_score(y_test[col], y_pred[:, i], average = 'weighted')
        recall = recall_score(y_test[col], y_pred[:, i],average = 'weighted')
        f1 = f1_score(y_test[col], y_pred[:, i], average = 'weighted')
        
        metrics.append([accuracy, precision, recall, f1])
    
    # store metrics
    metrics = np.array(metrics)
    data_metrics = pd.DataFrame(data = metrics, index = category_names, columns = ['Accuracy', 'Precision', 'Recall', 'F1'])
      
    return data_metrics

def save_model(model, model_filepath):
    """
    INPUT:
    model - ML model
    model_filepath - location to save the model
    
    OUTPUT:
    none
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

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