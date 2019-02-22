from nltk import punkt
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize 
import re

class LemmaTokenizer(object):
    
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    
    def __call__(self, articles):
        articles = re.sub('[^0-9a-zA-Z\'\x2F\x20\x3F\x40\x5F\xC0-\xFF]+', ' ', articles)
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]
