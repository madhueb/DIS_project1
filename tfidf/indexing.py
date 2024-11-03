from Tf_Idf import Tf_Idf_Vectorizer
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
    with open(args.token_dir+"/tokens_"+args.language+".pkl", "rb") as f:
        docs = pickle.load(f)

    tfidf = Tf_Idf_Vectorizer(max_df = 0.1, min_df = 0.01)
    tfidf.fit(docs)
    tfidf_matrix = tfidf.transform(docs)
    tfidf_matrix =tfidf_matrix.toarrray()
    tfidf_matrix = tfidf_matrix/np.linalg.norm(tfidf_matrix,axis=1)[:,None]
        
    #save tfidf model
    with open("tfidf"+args.lang+".pkl", "wb") as f:
        pickle.dump(tfidf, f)

    print("Tf-Idf model created for "+args.language)
    #Create Index 
    index = autofaiss.build_index(tfidf_matrix, save_on_disk=True, index_path="index"+args.language+".faiss", max_index_memory_usage="2GB",metric_type=faiss.METRIC_INNER_PRODUCT)

    print("Index Created for "+args.language)