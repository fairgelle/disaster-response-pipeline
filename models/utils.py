import re

import nltk 
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

def tokenize(text):
    '''
    Params:
      text - the text input to be tokenized.
      
    Returns:
      clean_tokens - cleaned up tokens after removal of stop words, lemmatization and tokenization
    
    '''
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    detected_urls = re.findall(url_regex, text) 
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
                
    tokens = word_tokenize(text)
    
    stop_words = stopwords.words("english")
    words = [t for t in tokens if t not in stop_words]
    
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for w in words:
        clean_word = lemmatizer.lemmatize(w).lower().strip()
        clean_tokens.append(clean_word)
        
    return clean_tokens