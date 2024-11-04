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

# LANGS = ["en", "fr", "de", "it", "es", "ar", "ko"]
LANGS = ["fr", "de", "it", "es", "ar", "ko"]
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
    spacy.cli.download(lang + "_core_news_sm")
    nlps[lang] = spacy.load(lang + "_core_news_sm")




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



def retrieve_top_k (queries, lang, batch_size=1000, k=10):

    tfidf = tfidfs[lang]
    # pos_doc = query["positive_docs"]

    queries = [preprocess_query(query, lang) for query in queries]
    if lang == "ar":
        all_tokens = []
        tokens_len = []
        for query in queries:
            tokens = simple_word_tokenize(query)
            all_tokens.extend(tokens)
            tokens_len.append(len(tokens))

        mle = MLEDisambiguator.pretrained()
        disambig = mle.disambiguate(all_tokens)

        tokens = []
        start = 0
        cnt = 0
        tmp = []
        for d in disambig:
            token = dediac_ar(d.analyses[0].analysis['lex']).translate(translator)
            if token not in ar_stop_words:
                tmp.append(token)
            cnt += 1
            if cnt == tokens_len[start]:
                tokens.append(tmp)
                tmp = []
                start += 1
                cnt = 0
    else:
        queries = nlps[lang].pipe(
            queries, batch_size=32, n_process=4
        )
        tokens = [
            [
                token.lemma_
                for token in query
                if not token.is_stop
                and not token.is_punct
                # and self.TOKEN_PATTERN.match(token.text)
            ]
            for query in tqdm(queries)
        ]
    # queries = [query for query in queries if query["lang"] == lang]
    #load tfidf model

    
    #transform query
    # query = tfidf.transform([query])
    # query = tfidf.transform([query], is_sparse=False)
    queries = tfidf.transform(tokens, is_query=True)

    # make queries as csr matrix
    q_values = torch.tensor(queries.data, dtype=torch.float32)
    q_crow_indices = torch.tensor(queries.indptr, dtype=torch.int32)
    q_col_indices = torch.tensor(queries.indices, dtype=torch.int32)

    # Create a PyTorch sparse_csr_tensor
    queries = queries.transpose()
    queries = torch.sparse_csr_tensor(q_crow_indices, q_col_indices, q_values, size=queries.shape, device=tfidf.device)
    #tfidf_matrix = tfidf.tfidf_matrix

    #Compute cosine similarity by batches :

    values = torch.tensor(tfidf.tfidf_matrix.data, dtype=torch.float32)
    crow_indices = torch.tensor(tfidf.tfidf_matrix.indptr, dtype=torch.int32)
    col_indices = torch.tensor(tfidf.tfidf_matrix.indices, dtype=torch.int32)

    # Create a PyTorch sparse_csr_tensor
    tfidf.tfidf_matrix = torch.sparse_csr_tensor(crow_indices, col_indices, values, size=tfidf.tfidf_matrix.shape, device=tfidf.device)

    top_k_index = tfidf.batch(queries, batch_size, k)

    # doc_ids = np.array([doc["docid"] for doc in documents if doc["lang"] == lang])
    # pos_doc_index = np.where(doc_ids == pos_doc)[0][0]
    # tfidf_matrix = tfidf.tfidf_matrix
    # print(pos_doc_index)
    #
    # print("similarity with positive doc : ", cosine_similarity(query, tfidf_matrix[pos_doc_index]))
    # for i in top_k_index:
    #     print("similarity with retrieved doc ", doc_ids[i], " : ", cosine_similarity(query, tfidf.tfidf_matrix[i]))


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

    queries = pd.read_csv(f'{args.token_dir}/train.csv')
    # # queries = queries[queries["lang"]=="fr"][:2]
    # queries = queries[queries["lang"].isin(LANGS)]
    # queries["doc_ids"] = queries.apply(retrieve_top_k, axis=1)
    # lang_accuracy = {lang: 0 for lang in LANGS}
    # for i, row in queries.iterrows():
    #     # if row["positive_docs"] in row["doc_ids"]:
    #     #     print("Document found in top 10")
    #     # else:
    #     #     print("Document not found in top 10")
    #     lang = row["lang"]
    #     if row["positive_docs"] in row["doc_ids"]:
    #         lang_accuracy[lang] += 1
    # lang_accuracy = {lang: acc / len(queries[queries["lang"] == lang]) for lang, acc in lang_accuracy.items()}
    # print(lang_accuracy)

    for lang in LANGS:
        queries_lang = queries[queries["lang"] == lang][["query", "positive_docs"]].reset_index(drop=True)
        doc_ids = retrieve_top_k(queries_lang["query"].tolist(), lang)
        acc = 0
        for i, row in queries_lang.iterrows():
            if row["positive_docs"] in doc_ids[i]:
                acc += 1
        print(f"Accuracy for {lang} : {acc / len(queries_lang)}")
        del tfidfs[lang]
        del nlps[lang]
        gc.collect()
        # clear memory


        # queries_lang.to_csv(f"{args.token_dir}/train_{lang}.csv", index=False)
        # print(f"Saved {lang} queries")