import numpy as np 
import pickle

import pandas as pd
import re
import spacy
import string

import torch
from camel_tools.utils.dediac import dediac_ar
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.disambig.mle import MLEDisambiguator
import nltk
import json
import argparse
from pathlib import Path
import gc

from tqdm import tqdm

punctuations = '''`÷×؛<>«»_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ''' + string.punctuation
with open('/nfs/scistore16/krishgrp/mansarip/Jupyter/DIS_project1/scripts/ar_stopwords.txt', 'r') as file:
    ar_stop_words = file.read().split('\n') + list(nltk.corpus.stopwords.words("arabic"))

translator = str.maketrans('', '', punctuations)

LANGS = ["fr", "en", "de", "it", "es", "ar", "ko"]
# LANGS = ["fr", "de", "it", "es", "ar", "ko"]

tfidfs = {}
print("cuda available : ", torch.cuda.is_available())
for lang in LANGS:
    with open(f"tfidf_{lang}.pkl", "rb") as f:
        tfidfs[lang] = pickle.load(f)
        tfidfs[lang].device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # tfidfs[lang].idf = tfidfs[lang].idf.to(tfidfs[lang].device)
        print(f"device for {lang} : {tfidfs[lang].device}")

# load doc ids dict with json
with open("/nfs/scistore16/krishgrp/mansarip/Jupyter/DIS_project1/data/ids_dict.json", "r") as f:
    ids_dict = json.load(f)

nlps = {}
for lang in LANGS:
    if lang == "ar":
        continue
    if lang == "en":
        spacy.cli.download("en_core_web_sm")
        nlps[lang] = spacy.load("en_core_web_sm")
    else:
        spacy.cli.download(lang + "_core_news_sm")
        nlps[lang] = spacy.load(lang + "_core_news_sm")

import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet

def synonym_expansion_nltk(query):
    expanded_query = set(query)

    for word in query:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                syss = lemma.name().replace("_", " ").lower().split()
                expanded_query.update(syss)

    return query + [word for word in expanded_query if word not in query]

def preprocess_query(query, lang):
    if lang == "ar":

        # Step 1: Remove URLs
        text = re.sub(r"http[s]?://\S+|www\.\S+", " ", query)

        # Step 2: Remove long sequences of non-alphanumeric characters (e.g., encoded data or code)
        text = re.sub(r"[^\w\s]{4,}", " ", text)  # Removes any sequence of 4 or more non-alphanumeric characters
        text = text.translate(translator)

        # Step 4: Remove excessive whitespace
        text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip().lower()

        # Step 5: Remove diacritics
        text = dediac_ar(text)

        return text

    else :
        # Step 1: Remove URLs
        text = re.sub(r"http[s]?://\S+|www\.\S+", " ", query)

        # Step 2: Remove long sequences of non-alphanumeric characters (e.g., encoded data or code)
        text = re.sub(r"[^\w\s]{4,}", " ", text)  # Removes any sequence of 4 or more non-alphanumeric characters

        # Step 3: Remove excessive whitespace
        text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip().lower()


        return text



def retrieve_top_k (tokens, lang, batch_size=1000, k=10):

    tfidf = tfidfs[lang]
    # # pos_doc = query["positive_docs"]
    #
    # queries = [preprocess_query(query, lang) for query in queries]
    # if lang == "ar":
    #     all_tokens = []
    #     tokens_len = []
    #     for query in queries:
    #         tokens = simple_word_tokenize(query)
    #         all_tokens.extend(tokens)
    #         tokens_len.append(len(tokens))
    #
    #     mle = MLEDisambiguator.pretrained()
    #     disambig = mle.disambiguate(all_tokens)
    #
    #     tokens = []
    #     start = 0
    #     cnt = 0
    #     tmp = []
    #     for d in disambig:
    #         token = dediac_ar(d.analyses[0].analysis['lex']).translate(translator)
    #         if token not in ar_stop_words:
    #             tmp.append(token)
    #         cnt += 1
    #         if cnt == tokens_len[start]:
    #             tokens.append(tmp)
    #             tmp = []
    #             start += 1
    #             cnt = 0
    # else:
    #     queries = nlps[lang].pipe(
    #         queries, batch_size=32, n_process=4
    #     )
    #     tokens = [
    #         [
    #             token.lemma_
    #             for token in query
    #             if not token.is_stop
    #             and not token.is_punct
    #             # and self.TOKEN_PATTERN.match(token.text)
    #         ]
    #         for query in tqdm(queries)
    #     ]
    # # tokens = [synonym_expansion_nltk(query) for query in tokens]

    queries = tfidf.transform(tokens, is_query=True)

    sims = (tfidf.tfidf_matrix @ queries.transpose()).toarray()
    top_k_index = np.argsort(sims.T, axis=1)[:, ::-1][:, :k]

    return np.array(ids_dict[lang])[top_k_index]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-dir", "--token_dir", type=Path, default = "./data")
    args = parser.parse_args()
    
    # #Load query :
    # queries = pd.read_csv("data/test.csv")
    # queries ["doc_ids"] = queries.apply(retrieve_top_k)
    # submission = queries[["id","doc_ids"]]
    # pd.to_csv("submission.csv",index=False)

    # with open(f'{args.token_dir}/corpus.json/corpus.json', "r") as f:
    #     documents = json.load(f)

    # queries = pd.read_csv(f'{args.token_dir}/train.csv')
    queries = pd.read_csv(f'{args.token_dir}/test.csv')


    ls = [[] for _ in range(len(queries))]
    queries["docids"] = ls
    for lang in LANGS:
        # queries_lang = queries[queries["lang"] == lang][["query", "positive_docs"]].reset_index(drop=True)
        queries_lang = queries[queries["lang"] == lang][["query"]].reset_index(drop=True)
        doc_ids = retrieve_top_k(queries_lang["query"].tolist(), lang)
        queries.loc[queries["lang"] == lang, "docids"] = pd.Series([doc_id.tolist() for doc_id in doc_ids], index=queries.loc[queries["lang"] == lang].index)

        # acc = 0
        # for i, row in queries_lang.iterrows():
        #     if row["positive_docs"] in doc_ids[i]:
        #         acc += 1
        # print(f"Accuracy for {lang} : {acc / len(queries_lang)}")
        gc.collect()


        # queries_lang.to_csv(f"{args.token_dir}/train_{lang}.csv", index=False)
        # print(f"Saved {lang} queries")
    queries = queries[["id", "docids"]]
    queries.to_csv(f"{args.token_dir}/submission.csv", index=False)