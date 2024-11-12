import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def download_nltk_resources():
    resources = ['punkt', 'stopwords', 'wordnet']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
            print(f"Successfully downloaded {resource}")
        except Exception as e:
           print(f"Error downloading {resource}: {str(e)}")

download_nltk_resources()

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re

number_pattern = re.compile(r'\d+')

crypto_terms = {'bitcoin', 'ethereum', 'blockchain', 'cryptocurrency'}
stop_words = set(stopwords.words('english')) - crypto_terms

def preprocess_text(text):
    if not text:
        return ""
    try:
        text = text.lower()
    
        text = number_pattern.sub('', text)
        
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        tokens = word_tokenize(text)
        
        tokens = [token for token in tokens if token not in stop_words]
        
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
        preprocessed_text = ' '.join(tokens)
        
        return preprocessed_text
    except Exception as e:
        print(f"Error preprocessing text: {e}")
        return text  

def preprocess_whitepaper(whitepaper_text):
    try:
      
        sentences = sent_tokenize(whitepaper_text)
        
        preprocessed_sentences = [preprocess_text(sentence) for sentence in sentences]
        
        preprocessed_whitepaper = ' '.join(preprocessed_sentences)
        
        return preprocessed_whitepaper
    except Exception as e:
        print(f"Error preprocessing whitepaper: {e}")
        return whitepaper_text 
 
