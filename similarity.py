import numpy as np
from numpy.linalg import norm

def calculate_similarity(embedding1, embedding2):
  
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    
    cosine_similarity = np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))
    
    return cosine_similarity
