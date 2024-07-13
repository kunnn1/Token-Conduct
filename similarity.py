import numpy as np
from numpy.linalg import norm

def calculate_similarity(embedding1, embedding2):
    """
    Calculate the cosine similarity between two embeddings.
    
    Args:
        embedding1 (numpy.ndarray): The first embedding vector.
        embedding2 (numpy.ndarray): The second embedding vector.
        
    Returns:
        float: The cosine similarity between the two embeddings.
    """
  
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    
    cosine_similarity = np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))
    
    return cosine_similarity

if __name__ == "__main__":
