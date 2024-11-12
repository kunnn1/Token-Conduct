# TokenConduct
My  project is a semantic similarity detector designed to identify and measure the similarity between different crypto white papers. The primary goal is to detect potential plagiarism and ensure the originality of academic and technical documents in the cryptocurrency domain, addressing the common fraud assoicated with white papers and fake ICOs. The detector uses natural language processing (NLP) techniques, specifically BERT embeddings, to calculate the semantic similarity between documents.
# Who Am I?
My name is Khasim Amedu and I am incoming freshman @ UIC and an aspiring AI/Machine Learning Engineer and Researcher. My Portfolio: https://kunnn1.github.io/
# Motivation
The project is a collaboration between my upperclassmen friends, Sam and Arnav who helped me lay out the edge cases and dependencies of this idea and drawing out the basic skeleton of what this tool would look like. Huge thanks to them!
# Appartus/Tech Used
Python

BERT (Bidirectional Encoder Representations from Transformers)

NLTK (Natural Language Toolkit)

Pandas

NumPy

Transformers (Hugging Face Library)

PyTorch
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

requirements.txt: Lists all the dependencies required for you to use the project. 

# How the Plagairism Detector Actually (Basically) Works
The detector intitally extracts text from inputted .txt file converting them into plain text that can be proccessed and analyzed. The extracted text is then cleaned and standardized by removing punctuation and numbers, converting to lowercase, eliminating stopwords, tokenization and lemmatization. This new preprocessed text is "fed" into a pre-trained BERT model, which generates numerical representations or embeddings that capture the semantic meaning of the text (roughly). The detector then compares the BERT embeddings of the input white paper against those of other documents in the crypto white paper database authroed by Sergi Valverde and Salva Duran-Nebreda (you can find on figshare :D) using cosine similarity to measure the semantic similarity. The white paper documents with cosine similarity scores above a predefined threshold of 0.9 are flagged as potential plagiarism cases, indicating a high degree of semantic similarity with the input white paper document. The output result of the program generates a report listing all flagged white papers and their similarity scores so that you can go back and review your white paper with the flagged ones.
# Drawbacks/Limits
The semantic similarity detector is pretty intensive, requiring significant processing power and time, especially for large datasets, which limits scalability and convenience as a tool for users. The testing phase of this building this detector adduced struggle with sophisticated paraphrasing, cross-lingual plagiarism, and nuanced domain-specific content, potentially leading to false positives. Additionally, the effectiveness depends on setting appropriate similarity thresholds and may require frequent updates to stay relevant as language evolves. Otherwise, it's pretty decent. 
