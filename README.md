# Disaster Response Classification Web App

## Table of Content
1. [Project Motivation](#project-motivation)
2. [Installation](#installation)
3. [About the Data](#about-the-data)
4. [Licensing, Authors & Acknowledgements](#licensing-authors--acknowledgements)

## Project Motivation
Climate change has increased the number and severity of natural disasters all over the world. The shorter the response time of emergency services, the greater the number of lives that can be saved. This app is using a machine learning model that was trained on over 25,000 natural-disaster- related messages. 

Disaster relief workers can use this app to input messages and it will return a classification of the message, such as "aid-related", "medical help", "water".
With better identification of messages, we hope that emergency services could allocate resources or assistance more effectively.

## Installation
The code should using Python version 3.6 and above. 

1. Start by [cloning](https://github.com/git-guides/git-clone) the repository into your local machine
2. The additional libraries required to execute the code can be installed by running `pip install -r requirements.txt`
3. Go ito the app/ directory and run `python run.py` to run the web app locally
4. Open http://0.0.0.0:3001/ on your browser to view the web app

## Code Overview
- Dataset: disaster_messages.csv and disaster_categories.csv contains messages data and the corresponding classification of the messages
- process_data.py: This script cleans up and merges the two dataset. The result is then saved into an SQLite DB. In this project, the DB is named as DisasterResponse DB. The whole process is executed by running `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
- train_classifier.py: This script loads the data from the DisasterResponse.db and uses it as an input for the ML model. The fine-tuned model is then saved as a pickle file called classifier.pkl. This whole process is executed by running `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
- templates folder: Contains files used for the frontend part of the web app
- run.py: A file that contains pre-processing of the data that will be passed into the files in the templates folder. This also contains the script necessary to render the web app.

## About the Data

###  Imbalanced Dataset

Some of the labels in the original dataset are highly imbalanced; this affects the model's F1 score. For simplicity's purposes, these highly imbalanced fields are dropped in the process_data.py file. In the future, pre-processing methods to deal with imbalanced dataset for classification such as [MLSMOTE](https://medium.com/thecyphy/handling-data-imbalance-in-multi-label-classification-mlsmote-531155416b87) can be used.

## Licensing, Authors & Acknowledgements
This project was completed as a part of Udacity Data Science Nanodegree. Credits to Figure Eight for providing the dataset.
