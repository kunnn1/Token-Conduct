import os
import pandas as pd

def load_whitepaper_database(directory):
    whitepapers = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                content = file.read()
                whitepapers.append({
                    'filename': filename,
                    'content': content
                })
    return pd.DataFrame(whitepapers)

whitepaper_dir = "/Users/khasimamedu/Downloads/whitepapers/txt_whitepapers_np"
whitepaper_df = load_whitepaper_database(whitepaper_dir)
