import hashlib
import datetime
import json
import os

global file_name

file_name = 'query_cache.json'

def cache_response(query, context, answer):
    query = query + "\n" + context
    hash_val = hashlib.sha256(query.encode('utf-8')).hexdigest()
    val = {
        hash_val : {
        "query": query,
        "answer": answer,
        "timestamp": datetime.datetime.today().strftime("%Y-%m-%dT%H:%M:%S")
        }
    }
    cur_files = os.listdir("/Users/vishaldawar/projects/rag_pipeline")
    if file_name in cur_files:
        with open(file_name, "r") as file:
            data = json.load(file)
        data.update(val)
    else:
        data = val
    with open(file_name, "w") as file:
        json.dump(data, file, indent=4)

def get_cached_response(query, context):
    query = query + "\n" + context
    hash_val = hashlib.sha256(query.encode('utf-8')).hexdigest()
    cur_files = os.listdir("/Users/vishaldawar/projects/rag_pipeline")
    if file_name in cur_files:
        with open(file_name, "r") as file:
            data = json.load(file)
    else:
        print("No Cache exists!")
        data = {}
    if hash_val in data.keys():
        return data[hash_val]['answer']
    else:
        return None
