import numpy as np 
import pickle
import faiss
import autofaiss
import pandas as pd
import re
import spacy
import string
from camel_tools.utils.dediac import dediac_ar
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.disambig.mle import MLEDisambiguator
import nltk
import json
import argparse
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from Tf_Idf import Tf_Idf_Vectorizer




def preprocess_query(query):
    lang = query["lang"]
    if lang == "ar":
        punctuations = '''`÷×؛<>«»_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ''' + string.punctuation
        with open('/nfs/scistore16/krishgrp/mansarip/Jupyter/DIS_project1/scripts/ar_stopwords.txt', 'r') as file:
            stop_words = file.read().split('\n') + list(nltk.corpus.stopwords.words("arabic"))


        # Step 1: Remove URLs
        text = re.sub(r"http[s]?://\S+|www\.\S+", " ", query["query"])

        # Step 2: Remove long sequences of non-alphanumeric characters (e.g., encoded data or code)
        text = re.sub(r"[^\w\s]{4,}", " ", text)  # Removes any sequence of 4 or more non-alphanumeric characters
        translator = str.maketrans('', '', punctuations)
        text = text.translate(translator)

        # Step 4: Remove excessive whitespace
        text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip().lower()

        # Step 5: Remove diacritics
        text = dediac_ar(text)

        # Step 6: Tokenize text
        tokens = simple_word_tokenize(text)

        # Step 7: Lemmatize each token and remove stopwords
        mle = MLEDisambiguator.pretrained() 
        disambig = mle.disambiguate(tokens)

        lemmas = [dediac_ar(d.analyses[0].analysis['lex']).translate(translator) for d in disambig]
        lemmas = [lemma for lemma in lemmas if lemma not in stop_words]
        return " ".join(lemmas)

    else :
        # Step 1: Remove URLs
        text = re.sub(r"http[s]?://\S+|www\.\S+", " ", query["query"])

        # Step 2: Remove long sequences of non-alphanumeric characters (e.g., encoded data or code)
        text = re.sub(r"[^\w\s]{4,}", " ", text)  # Removes any sequence of 4 or more non-alphanumeric characters

        # Step 3: Remove excessive whitespace
        text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip().lower()

        # Step 4: Tokenize

        nlp = spacy.load(lang+"_core_news_sm")
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        return " ".join(tokens)
    

lang = "fr"
with open("tfidf"+lang+".pkl", "rb") as f:
    tfidf = pickle.load(f)



def retrieve_top_k (query,k=10):
    lang = query["lang"]
    pos_doc = query["positive_docs"]

    query = preprocess_query(query)
    #load tfidf model

    
    #transform query
    query = tfidf.transform([query])
    #tfidf_matrix = tfidf.tfidf_matrix

    #Compute cosine similarity by batches :

    top_k_index = tfidf.batch(tfidf.tfidf_matrix, query, 1000, 10)

    doc_ids = np.array([doc["docid"] for doc in documents if doc["lang"] == lang] )
    pos_doc_index = np.where(doc_ids == pos_doc)[0][0]
    tfidf_matrix =tfidf.tfidf_matrix
    print(pos_doc_index)

    print("similarity with positive doc : ", cosine_similarity(query, tfidf_matrix[pos_doc_index]))
    for i in top_k_index:  
        print("similarity with retrieved doc ", doc_ids[i], " : ", cosine_similarity(query, tfidf.tfidf_matrix[i]))
    return doc_ids[top_k_index]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-dir", "--token_dir", type=Path, default = "./data")
    args = parser.parse_args()
    
    # #Load query :
    # queries = pd.read_csv("data/test.csv")
    # queries ["doc_ids"] = queries.apply(retrieve_top_k)
    # submission = queries[["id","doc_ids"]]
    # pd.to_csv("submission.csv",index=False)

    with open(f'{args.token_dir}/corpus.json/corpus.json', "r") as f:
        documents = json.load(f)

    queries = pd.read_csv(f'{args.token_dir}/dev.csv')
    queries = queries[queries["lang"]=="fr"][:2]
    queries["doc_ids"] = queries.apply(retrieve_top_k, axis=1)
    print(queries)
    for i, row in queries.iterrows():
        if row["positive_docs"] in row["doc_ids"]:
            print("Document found in top 10")
        else:
            print("Document not found in top 10")

