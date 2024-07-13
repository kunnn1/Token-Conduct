import os

def extract_text_from_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_text_from_directory(directory):
    whitepapers = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            text = extract_text_from_txt(filepath)
            whitepapers.append({
                'filename': filename,
                'content': text
            })
    return whitepapers
