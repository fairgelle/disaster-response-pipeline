import numpy as np
import pandas as pd

from sqlalchemy.engine import create_engine
import sys


def load_data(messages_filepath, categories_filepath):
    '''
    Returns a df that is a result of the merged 'messages' &
    'categories' data. The data are concatenated by the id.
    
    Args:
        messages_filepath (string): The filepath of the messages csv.
        param2 (str): The filepath of the categories csv.
    
    Returns:
        df (dataframe): The resulting dataframe of the concatenation of the
            two datasets.
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = pd.merge(messages, categories, on='id')
    
    return df


def clean_data(df):
    '''
    Cleans the dataframe by splitting up the categories column values
    into separate columns. 
    Duplicates will be removed here as well.
    
    Args:
        df (dataframe): Raw dataframe.
    
    Returns:
        df (dataframe): Cleaned dataframe
    '''

    # excluding columns with imbalance dataset for simplicity's sake
    cols_to_exclude = [
        'child_alone', 'security', 'offer', 'money', 'missing_people', 'buildings',
        'other_weather', 'fire', 'other_infrastructure', 'infrastructure_related',
        'other_aid', 'aid_centers', 'shops', 'hospitals', 'tools', 'electricity'
    ]
    
    categories = df.categories.str.split(pat=';', expand=True)
    row = categories.iloc[0]

    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = pd.to_numeric(categories[column])
        
    df = df.drop(['categories'], axis=1)
    
    df = pd.concat([df, categories], axis=1)
    
    df = df.drop(cols_to_exclude, axis=1).drop_duplicates()
    
    df['related'] = df['related'].replace(2, 1)
    
    return df


def save_data(df, database_filename):
    '''
    Saves data into SQLite DB.
    
    Args:
        df (dataframe): The dataframe to be saved into the DB
        database_filename (string): The resulting name of the table that is 
            saved in SQLite db. The DB name and the table name would be the same
    '''
    engine = create_engine('sqlite:///{db_name}'.format(db_name=database_filename))
    df.to_sql(database_filename, engine, index=False, if_exists='replace') 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()