import re
import nltk
from nltk.tokenize import word_tokenize
from typing import List
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
 
nltk.download('stopwords')
nltk.download('punkt')
wnl = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text: str) -> List[str]:
    '''Preprocess a text for spam classification.

    Returns
    -------
    List[str]
        that contains the preprocessed words of the text.
    '''
    tokens = tokenize_text(text)
    tokens = [ remove_non_english_letters(token) for token in tokens ]
    tokens = [ wnl.lemmatize(token) for token in tokens ]
    tokens = [ token for token in tokens if not token in stop_words ]
    # This last one removes empty strings
    tokens = [ token for token in tokens if token ]

    return tokens

def tokenize_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens]
    return tokens


def remove_non_english_letters(input_string):

    pattern = re.compile('[^a-zA-Z]')
    return pattern.sub('', input_string)