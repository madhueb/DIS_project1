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

with open("./Data/corpus.json/corpus.json", "r") as f:
    documents = json.load(f)


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
    
    

def retrieve_top_k (query,k=10):
    
    lang = query["lang"]
    query = preprocess_query(query)
    #load tfidf model
    with open("tfidf"+lang+".pkl", "rb") as f:
        tfidf = pickle.load(f)
    
    #transform query
    query = tfidf.transform([query])

    #Compute cosine similarity by batches :

    top_k_index = tfidf.batch(tfidf, query, 1000, 10)

    doc_ids = [doc["doc_ids"] for doc in documents if doc["lang"] == lang]  

    return doc_ids[top_k_index]


if __name__ == "__main__":
    
    # #Load query :
    # queries = pd.read_csv("data/test.csv")
    # queries ["doc_ids"] = queries.apply(retrieve_top_k)
    # submission = queries[["id","doc_ids"]]
    # pd.to_csv("submission.csv",index=False)

    queries = pd.read_csv("data/dev.csv")
    queries ["doc_ids"] = queries.apply(retrieve_top_k)
    for i, row in queries.iterrows():
        if row["positive_docs"] in row["doc_ids"]:
            print("Document found in top 10")
        else:
            print("Document not found in top 10")