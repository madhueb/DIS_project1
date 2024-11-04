from Tf_Idf import Tf_Idf_Vectorizer

import pickle 

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

    
    tfidf = Tf_Idf_Vectorizer()
    tfidf.fit_transform(docs)

    #save tfidf model
    with open(f"tfidf_{args.language}.pkl", "wb") as f:
        pickle.dump(tfidf, f)


    print("Tf-Idf model created for " + args.language)
