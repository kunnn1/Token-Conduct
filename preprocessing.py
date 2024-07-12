import nltk
import ssl
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Handle SSL certificate verification
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Function to download NLTK resources
def download_nltk_resources():
    resources = ['punkt', 'stopwords', 'wordnet']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
            logger.info(f"Successfully downloaded {resource}")
        except Exception as e:
            logger.error(f"Error downloading {resource}: {str(e)}")

# Download NLTK resources
download_nltk_resources()

# Now import the required NLTK modules
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re

# Compile regex pattern for number removal
number_pattern = re.compile(r'\d+')

# Define crypto-related terms to keep
crypto_terms = {'bitcoin', 'ethereum', 'blockchain', 'cryptocurrency'}
stop_words = set(stopwords.words('english')) - crypto_terms

def preprocess_text(text):
    if not text:
        return ""
    try:
        # Convert to lowercase
        text = text.lower()
        
        # Remove numbers
        text = number_pattern.sub('', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in stop_words]
        
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        # Join tokens back into text
        preprocessed_text = ' '.join(tokens)
        
        return preprocessed_text
    except Exception as e:
        logger.error(f"Error preprocessing text: {e}")
        return text  # Return original text if preprocessing fails

def preprocess_whitepaper(whitepaper_text):
    try:
        # Split the whitepaper into sentences
        sentences = sent_tokenize(whitepaper_text)
        
        # Preprocess each sentence
        preprocessed_sentences = [preprocess_text(sentence) for sentence in sentences]
        
        # Join preprocessed sentences back into a single text
        preprocessed_whitepaper = ' '.join(preprocessed_sentences)
        
        return preprocessed_whitepaper
    except Exception as e:
        logger.error(f"Error preprocessing whitepaper: {e}")
        return whitepaper_text  # Return original text if preprocessing fails

if __name__ == "__main__":
    # Example usage
    sample_texts = [
        "Bitcoin is a decentralized digital currency.",
        "Ethereum introduced smart contracts to blockchain.",
        "",  # Test empty input
        "123 numbers and punctuation!!! should be removed.",
    ]
    for text in sample_texts:
        preprocessed = preprocess_whitepaper(text)
        print("Original:", text)
        print("Preprocessed:", preprocessed)
        print()
