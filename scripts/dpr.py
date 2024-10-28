import argparse
import logging
import os
import json

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    DPRContextEncoder,
    DPRContextEncoderTokenizerFast,
    RagRetriever,
    RagSequenceForGeneration,
    RagTokenizer,
)

from datasets import load_dataset

from model.custom_utils.model import get_prompt_format, options_to_text, embed
from model.custom_utils.benchmark import run_benchmark, get_compute_metrics
import random
import faiss
from optimum.gptq import GPTQQuantizer


from functools import partial

import torch.nn as nn
import faiss
from datasets import load_dataset
from transformers import (
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizerFast,
    DPRReader,
    DPRReaderTokenizerFast,
)

class DPRModule(nn.Module):
    '''
    Custom module for DPR
    '''
    def __init__(self, config, **rag_module_args):
        super(DPRModule, self).__init__()
        # Get optional kwargs
        self.k = rag_module_args.get('n_docs', 1)

        # Get Question Encoder and Tokenizer
        self.q_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained(rag_module_args["encoder_model_path"])
        self.q_encoder = DPRQuestionEncoder.from_pretrained(rag_module_args["encoder_model_path"])

        # Get documents
        self.docs = load_dataset(rag_module_args.get("document_path", "wiki_dpr"), split='train')

        # Retriever Index
        self.docs.add_faiss_index("embeddings")

    def forward(self, question):
        # TODO: Perform forward pass
        input_ids = self.q_tokenizer(question, return_tensors="pt")
        q_embed = self.q_encoder(**input_ids).pooler_output.detach().numpy()

        # Retrieve nearest examples
        scores, context = self.docs.get_nearest_examples("embeddings", q_embed, k=self.k)

        return context['text']


if __name__ == '__main__':
    def dpr_pipeine(args):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load context encoder and tokenizer
        ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(args.model)
        ctx_encoder = DPRContextEncoder.from_pretrained(args.model).to(device=device)

        # Load dataset
        split = f"train[:{args.N}]" if args.N else "train"
        ds = load_dataset(args.dataset, args.subset, split=split)

        # Reformat
        if args.dataset == 'thewordsmiths/wiki_stem':
            ds = ds.rename_columns({'content_id': 'id', 'page_title': 'title'})

        # Embedd
        ds = ds.map(
            partial(embed, ctx_encoder=ctx_encoder, ctx_tokenizer=ctx_tokenizer, device=device),
            batched=True,
            batch_size=args.batch_size,
        )

        # Keep only useful columns
        ds = ds.select_columns(['id', 'text', 'title', 'embeddings'])

        # Push dataset to hub
        if args.subset:
            ds.push_to_hub(f"{HF_ORGANIZATION}/stem_dpr", args.subset, token=HF_TOKEN)
        else:
            ds.push_to_hub(f"{HF_ORGANIZATION}/stem_dpr", token=HF_TOKEN)

        # FAISS Index
        index = faiss.IndexHNSWFlat(768, 128, faiss.METRIC_INNER_PRODUCT)
        ds.add_faiss_index("embeddings", custom_index=index)

        # Get retriever and tokenizer
        tokenizer = RagTokenizer.from_pretrained(args.model)
        retriever = RagRetriever.from_pretrained(args.model, index_name="custom", indexed_dataset=ds)
        model = RagSequenceForGeneration.from_pretrained(args.model, retriever=retriever)

        # Push retriever to hub
        model.push_to_hub(f"{HF_ORGANIZATION}/stem_rag", token=HF_TOKEN)
        tokenizer.push_to_hub(f"{HF_ORGANIZATION}/stem_retriever", token=HF_TOKEN)
