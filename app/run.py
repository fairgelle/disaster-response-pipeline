import json
import plotly
import pandas as pd
import re
import joblib

import nltk 
import sys
import os
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords']) #delete

# from nltk.tokenize import word_tokenize #delete
# from nltk.stem import WordNetLemmatizer #delete
# from nltk.corpus import stopwords #delete

# testing
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.join(FILE_DIR, os.pardir) 
MODEL_DIR = os.path.join(PARENT_DIR, 'models')

sys.path.append(MODEL_DIR)
#sys.path.append(os.path.relpath('../models'))
from utils import tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine


app = Flask(__name__)


# load data
#engine = create_engine('sqlite:///../data/DisasterResponse.db')
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('data/DisasterResponse.db', engine)

# load model
model = joblib.load("models/classifier.pkl")
#model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)


    # proportion of class with value 1
    class_prop_1 = (df.drop(['id', 'message', 'original', 'genre'], axis = 1).sum() / 
      len(df)).sort_values(ascending=False)

    # proportion of class with value 0
    class_prop_0 = 1 - class_prop_1
    
    # extracting the label of the class
    class_labels = list(class_prop_1.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    marker=dict(
                      color='rgba(0, 105, 146)'
                    ),
                )
            ],
            
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        ####
        {
            'data': [
                Bar(
                    x=class_labels,
                    y=class_prop_1,
                    name = 'Class = 1',
                    marker = dict(
                      color = 'rgb(0, 105, 146)'
                    )
                    #orientation = 'h'
                ),
                Bar(
                    x=class_labels,
                    y=class_prop_0,
                    name = 'Class = 0',
                    marker = dict(
                      color = 'rgb(236, 164, 0)'
                    )
                    #orientation = 'h'
                )
            ],

            'layout': {
                'title': 'Distribution of labels within classes',
                'yaxis': {
                    'title': "Distribution"
                },
                'xaxis': {
                    'title': "Class",
            #        'tickangle': -45
                },
                'barmode' : 'stack'
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()