import re
import string
import nltk
import pandas as pd
import logging
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def remove_tags(text):
    p = re.compile(r'<.*?>')
    return p.sub('', text)

def remove_url(txt):
    return re.sub(r'\s*https?://\S+(\s+|$)', '', txt, flags=re.MULTILINE)

def remove_hashtag(txt):
    return re.sub(r'@[A-Za-z0-9]+', '', txt, flags=re.MULTILINE)

def remove_punc(txt):
    return txt.translate(str.maketrans('', '', string.punctuation))

def remove_spchar(text):
    return re.sub('\W+',' ', text)

def remove_nonascii(txt):
    encoded_string = txt.encode("ascii", "ignore")
    return encoded_string.decode()

def remove_stopwords(txt):
    data = txt.split()
    new = [word for word in data if not word in stopwords.words('english')]
    return " ".join(new)

def lematize(text):
    wnl = WordNetLemmatizer()
    return " ".join([wnl.lemmatize(i,pos='v') for i in text.split()])

def preprocess_df(df):
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

    logging.info("Downloaded all 3 libraries")

    # Lower all caps
    df['text'] = df['text'].apply(lambda x: x.lower())

    # Remove Html Tags
    df['text'] = df['text'].apply(lambda x: remove_tags(x))

    # Remove links:
    df['text'] = df['text'].apply(lambda x: remove_url(x))

    # Remove hash tag:
    df['text'] = df['text'].apply(lambda x: remove_hashtag(x))

    # Remove punctuation
    df['text'] = df['text'].apply(lambda x: remove_punc(x))

    # Remove Special Char
    df['text'] = df['text'].apply(lambda x: remove_spchar(x))

    #Remove Non- ascii characters
    df['text'] = df['text'].apply(lambda x: remove_nonascii(x))

    logging.info("Above all preprocessing done. Starting stopwords")

    #Remove Stopwords
    df['text'] = df['text'].apply(lambda x: remove_stopwords(x))

    logging.info("Stopwords done. Starting lemmatization")
    
    # Applying lemmatization
    df['review'] = df['review'].apply(lambda x: lematize(x))

    logging.info("Preprocessing completed")

    return df

