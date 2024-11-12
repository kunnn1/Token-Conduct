# TokenConduct
TokenConduct is my semantic plagiarism detection tool that leverages B.E.R.T. sentence vector embeddings from Google's B.E.R.T encoder Transformers architecture to detect potential cases of semantic plagairsim of fradulent/recycled ICO whitepapers, a common form of cryptocurrency fraud. I processed a Figshare database of over 1,300+ genuine ICO whitepapers from various cryptocurrencies and blockchains such as Bitcoin and EtheruemX using pre-processing techiques such as B.E.R.T. tokenization, lemmatization and stop-word removal to clean the data for the uncased B.E.R.T. model. From there, the tokenized embeddings are fed into the B.E.R.T architecture and yield high-dimensional bidrectional contextualized embeddings that encode the essential semantic information of the database of genuine ICO whitepapers, and compares the input ICO whitepaper vector embedding to over 1,300+ crypto whitepapers using cosine similarity with a semantic similarity parametric threshold of 95% and scores the potential semantic plagairism of your input ICO whitepaper with the result of such cosine similarity. 
# Who Am I?
My name is Khasim Amedu and I am a first-year Computer Science and Linguistics major @ UIC and an aspiring Machine Learning Engineer and Quantitative Developer/Trader. My Portfolio: https://kunnn1.github.io/
# Motivation
The project is a collaboration between me and my upperclassmen friends, Sam and Arnav who helped me lay out the edge cases and dependencies of this idea and drawing out the basic skeleton of what this tool would look like. Huge thanks to them!
# Appartus
- Python

- B.E.R.T. (Transformers)

- NLTK (Natural Language Toolkit)

- Pandas

- NumPy

- PyTorch
# How to Install?
1. Clone this repository:
   
```git clone https://github.com/yourusername/semantic-similarity-detector.git``` and 
```cd semantic-similarity-detector```

2. Create a virtual environment:
   
```python -m venv venv```
```# On Mac: source venv/bin/activate  # On Windows: venv\Scripts\activate```

3. Install the required dependicies:
   
```pip install -r requirements.txt```

4. Download necessary NLTK data:
   
```import nltk nltk.download('punkt') nltk.download('stopwords') nltk.download('wordnet')```
# How to Use?
1. Prepare the white paper documents you want to use, and make sure they are in .txt format.
2. Run the main script (main.py)   
```python main.py```
3. Input your white paper in main.py by modifying the ```input_file_path``` variable with your test document.
4. View the cosine similarity scores in the output. 


# Project Structure
main.py: The main script and logic to run the plagiarism detection.

preprocessing.py: Uses functions for text preprocessing.

model.py: Uses functions to generating BERT embeddings.

similarity.py: Contains functions for calculating similarity between embeddings.

whitepaper_database.py: Manages the database of white papers.

extract_text.py: Uses functions to extract text from files.

downloads.py: Handles downloading necessary resources or datasets.

# Drawbacks/Limits
The semantic similarity detector is pretty intensive, requiring significant processing power and time, especially for large datasets, which limits scalability and convenience as a tool for users. The testing phase of this building this detector adduced struggle with sophisticated paraphrasing, cross-lingual plagiarism, and nuanced domain-specific content, potentially leading to false positives. Additionally, the effectiveness depends on setting appropriate similarity thresholds and may require frequent updates to stay relevant as language evolves. Otherwise, it's pretty decent. 
