import pandas as pd
from pypdf import PdfReader
import os

files = os.listdir("./data")
files = [x for x in files if '.ipynb' not in x]

def read_data(files):
    all_text = ""
    for file_name in files:
        if '.csv' in file_name:
            data = pd.read_csv(f"./data/{file_name}")
            texts = [", ".join([f"{col1}: {row[col1]}" for col1 in data.columns]).strip() for idx, row in data.iterrows()]
            texts = "\n".join(texts)
            all_text = all_text + texts
        elif '.pdf' in file_name:
            reader = PdfReader(f"./data/{file_name}")
            data = ''        

            # Loop through every page in the document
            for index, page in enumerate(reader.pages):
                # Extract clean text from the specific page
                text = page.extract_text()
                data = data + text
            all_text = all_text + data
        else:
            raise ValueError("RAG Pipeline can only ingest data from csv or pdf files!")
    return all_text
    

if __name__ == "__main__":
    files = os.listdir("./data")
    files = [x for x in files if '.ipynb' not in x]
    data = read_data(files)
    print(data)