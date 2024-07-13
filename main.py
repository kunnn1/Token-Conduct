import os
import logging
from difflib import SequenceMatcher
from extract_text import extract_text_from_directory
from preprocessing import preprocess_text
from model import get_bert_embedding
from similarity import calculate_similarity
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess_whitepapers(directory):
    logger.info("Loading and preprocessing whitepapers...")
    whitepapers = extract_text_from_directory(directory)
    for paper in whitepapers:
        paper['content'] = preprocess_text(paper['content'])
    return pd.DataFrame(whitepapers)

def detect_plagiarism(input_paper, whitepaper_df, threshold=0.9):
    logger.info("Detecting plagiarism...")
    try:
        input_embedding = get_bert_embedding(preprocess_text(input_paper))
        potential_plagiarism = []

        for i, (_, paper) in enumerate(whitepaper_df.iterrows()):
            logger.info(f"Processing paper {i+1}/{len(whitepaper_df)}")
            paper_embedding = get_bert_embedding(paper['content'])
            similarity = calculate_similarity(input_embedding, paper_embedding)
            if similarity > threshold:
                potential_plagiarism.append((paper['filename'], similarity))

        return potential_plagiarism
    except Exception as e:
        logger.error(f"Error in detect_plagiarism: {str(e)}")
        raise

def main():
    logger.info("Starting main function...")
    whitepaper_dir = "/Users/khasimamedu/Downloads/whitepapers/txt_whitepapers_np"
    input_file_path = "your_input_whitepaper"

    if not os.path.exists(whitepaper_dir):
        logger.error(f"Whitepaper directory '{whitepaper_dir}' does not exist.")
        return
    if not os.path.isfile(input_file_path):
        logger.error(f"Input file '{input_file_path}' does not exist.")
        return

    try:
        whitepaper_df = load_and_preprocess_whitepapers(whitepaper_dir)
        
        with open(input_file_path, 'r', encoding='utf-8') as file:
            input_paper = file.read()
        
        results = detect_plagiarism(input_paper, whitepaper_df, threshold=0.9)

        for paper, similarity in results:
            logger.info(f"Potential plagiarism detected:")
            logger.info(f"Similarity: {similarity}")
            logger.info(f"Matched paper: {paper}\n")
    except Exception as e:
        logger.error(f"An error occurred in main: {str(e)}")

if __name__ == "__main__":
    main()
