from ingest import read_data
import tiktoken
import os


def get_chunks(all_text):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(all_text)  # convert full text to tokens first
    
    all_chunks = []
    i, idx = 0, 0
    
    while i < len(tokens):        # iterate over tokens, not characters
        token_chunk = tokens[i:i+300]
        text = enc.decode(token_chunk)  # convert back to string
        all_chunks.append({"text": text, "chunk_index": f"chunk_{idx}"})
        i += 250
        idx += 1
    
    return all_chunks

if __name__ == "__main__":
    files = os.listdir("./data")
    files = [x for x in files if '.ipynb' not in x]
    all_text = read_data(files)

    all_chunks = get_chunks(all_text)
    for idx, value in enumerate(all_chunks):
        print(value)

    print(len(all_chunks))
