
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

# from tokenization.de.de_tokenizers import GermanTokenizerV2

punctuations = '''`÷×؛<>«»_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ''' + string.punctuation
with open('/nfs/scistore16/krishgrp/mansarip/Jupyter/DIS_project1/scripts/ar_stopwords.txt', 'r') as file:
    ar_stop_words = file.read().split('\n') + list(nltk.corpus.stopwords.words("arabic"))

translator = str.maketrans('', '', punctuations)

LANGS = ["fr", "en", "de", "it", "es", "ar", "ko"]
# LANGS = ["fr", "de", "it", "es", "ar", "ko"]


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



def get_tokens(queries, lang):

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
    # if lang == "de":
    #     tokenizer = GermanTokenizerV2()
    #     tokens = tokenizer.tokenize_batch(queries, batch_size=32, n_process=4)
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

    return tokens
