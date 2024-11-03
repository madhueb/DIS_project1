from Tf_Idf import Tf_Idf_Vectorizer
import numpy as np
import pickle 
import autofaiss
import faiss
import argparse


if __name__ == "__main__":
    #arguments 

    parser = argparse.ArgumentParser()

    parser.add_argument("-dir", "--token_dir", type = str, default = "./data/")
    args = parser.parse_args()


    languages = ['ar','de','en','es','fr','it','ko']

    for lang in languages:
        #Load documents tokenization  : 
        with open(args.token_dir+"/tokens_"+lang+".pkl", "rb") as f:
            docs = pickle.load(f)

        tfidf = Tf_Idf_Vectorizer(max_df = 0.1, min_df = 0.01)
        tfidf.fit(docs)
        tfidf_matrix = tfidf.transform(docs)
        tfidf_matrix =tfidf_matrix.toarrray()
        tfidf_matrix = tfidf_matrix/np.linalg.norm(tfidf_matrix,axis=1)[:,None]
        
        #save tfidf model
        with open("tfidf"+lang+".pkl", "wb") as f:
            pickle.dump(tfidf, f)

        #Create Index 
        index = autofaiss.build_index(tfidf_matrix, save_on_disk=True, index_path="index"+lang+".faiss", max_index_memory_usage="2GB",metric_type=faiss.METRIC_INNER_PRODUCT)

        print("Index Created for "+lang)