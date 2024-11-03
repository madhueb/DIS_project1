from Tf_Idf import Tf_Idf_Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle 
import autofaiss
import faiss
import argparse
from pathlib import Path



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-dir", "--token_dir", type=Path, default = "./data")
    parser.add_argument("-lang", "--language", type=str, required=True, choices=['ar','de','en','es','fr','it','ko'])
    args = parser.parse_args()

   
    #Load documents tokenization  : 
    with open(f'{args.token_dir}/tokens_{args.language}.pkl', "rb") as f:
        docs = pickle.load(f)

    docs = [" ".join(doc) for doc in docs]

    
    tfidf = Tf_Idf_Vectorizer(max_df = 0.1, min_df = 0.01)
    tfidf.fit_transform(docs)

    #save tfidf model
    with open("tfidf"+args.language+".pkl", "wb") as f:
        pickle.dump(tfidf, f)


    print("Tf-Idf model created for "+args.language)
    # #Create Index 
    # index = autofaiss.build_index(tfidf_matrix, save_on_disk=True, index_path="index"+args.language+".faiss", max_index_memory_usage="2GB",metric_type=faiss.METRIC_INNER_PRODUCT)

    # print("Index Created for "+args.language)