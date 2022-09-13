import argparse
import logging
import os
import math
from dataclasses import dataclass, field
import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from torch import Tensor

from transformers.models.mbart.modeling_mbart import MBartLearnedPositionalEmbedding
from transformers.models.bart.modeling_bart import BartLearnedPositionalEmbedding
from transformers import MBartForConditionalGeneration, MBartConfig, MBart50Tokenizer
from transformers import PreTrainedTokenizerFast
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers.models.longformer.modeling_longformer import LongformerSelfAttention
from transformers.models.bart.modeling_bart import BartLearnedPositionalEmbedding

import warnings
warnings.filterwarnings(action='ignore')

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LongformerSelfAttentionForMBart(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.embed_dim = config.d_model
        self.longformer_self_attn = LongformerSelfAttention(
            config, layer_id=layer_id)
        self.output = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        attention_mask = attention_mask.squeeze(dim=1)
        attention_mask = attention_mask[:, 0]

        is_index_masked = attention_mask < 0
        is_index_global_attn = attention_mask > 0
        is_global_attn = is_index_global_attn.flatten().any().item()

        outputs = self.longformer_self_attn(
            hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=None,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=output_attentions,
        )

        attn_output = self.output(outputs[0])

        return (attn_output,) + outputs[1:] if len(outputs) == 2 else (attn_output, None, None)


class LongformerEncoderDecoderForConditionalGeneration(MBartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

        if config.attention_mode == 'n2':
            pass  # do nothing, use BertSelfAttention instead
        else:

            self.model.encoder.embed_positions = MBartLearnedPositionalEmbedding(
                config.max_encoder_position_embeddings,
                config.d_model)

            self.model.decoder.embed_positions = MBartLearnedPositionalEmbedding(
                config.max_decoder_position_embeddings,
                config.d_model)

            for i, layer in enumerate(self.model.encoder.layers):
                layer.self_attn = LongformerSelfAttentionForMBart(
                    config, layer_id=i)


class LongformerEncoderDecoderConfig(MBartConfig):
    def __init__(self, attention_window: List[int] = None, attention_dilation: List[int] = None,
                 autoregressive: bool = False, attention_mode: str = 'sliding_chunks',
                 gradient_checkpointing: bool = False, **kwargs):
        """
        Args:
            attention_window: list of attention window sizes of length = number of layers.
                window size = number of attention locations on each side.
                For an affective window size of 512, use `attention_window=[256]*num_layers`
                which is 256 on each side.
            attention_dilation: list of attention dilation of length = number of layers.
                attention dilation of `1` means no dilation.
            autoregressive: do autoregressive attention or have attention of both sides
            attention_mode: 'n2' for regular n^2 self-attention, 'tvm' for TVM implemenation of Longformer
                selfattention, 'sliding_chunks' for another implementation of Longformer selfattention
        """
        super().__init__(**kwargs)
        self.attention_window = attention_window
        self.attention_dilation = attention_dilation
        self.autoregressive = autoregressive
        self.attention_mode = attention_mode
        self.gradient_checkpointing = gradient_checkpointing
        assert self.attention_mode in ['tvm', 'sliding_chunks', 'n2']

tokenizer = MBart50Tokenizer.from_pretrained("./results/mbart-long/checkpoint-1100")

model = LongformerEncoderDecoderForConditionalGeneration.from_pretrained("./results/mbart-long/checkpoint-1100/")

import json
from pathlib import Path
from sys import getsizeof
from memory_profiler import profile

# pympler
from pympler.tracker import SummaryTracker
tracker = SummaryTracker()

filenames = []
def read_dataset(dir):
    dir = Path(dir)
    texts = []  # celoten tekst naloge (z naslovi)
    abstracts = []

    for i, json_file in enumerate(dir.iterdir()):
        # print(f'i: {i}, file: {json_file}')
        filenames.append(json_file)
        
        # ustvarjanje teksta
        text = ""
        with open(json_file, 'r') as f:
            slovar = json.load(f)

            for key in slovar.keys():
                if key == "abstract":
                    continue

                # vsak odsek ima pododseke
                for section in slovar[key]:
                    text += f" {section}"
                    text += '\n'

            texts.append(text)
            abstracts.append(slovar["abstract"][0])

    return texts, abstracts

# what split of short summaries to use
split = "tokenized_bottom_10"

texts, abstracts = read_dataset(f"./data/short-summaries/{split}")

# assign 80% of examples to train split and 20% to test
# originally first 80% was used for training
eighty_percent = int(len(texts) * 0.8)


EVAL_SIZE = 1100

test_filenames = filenames[eighty_percent:eighty_percent+EVAL_SIZE]
test_texts = texts[eighty_percent:eighty_percent+EVAL_SIZE]
test_abstracts = abstracts[eighty_percent:eighty_percent+EVAL_SIZE]

# define function for summarization
max_seq_len = 9216

def summarize(text, max_len):

    context_tokens = ['<s>'] + tokenizer.tokenize(text) + ['</s>']
    input_ids = tokenizer.convert_tokens_to_ids(context_tokens) 

    if len(input_ids) < max_seq_len:   
            while len(input_ids) < max_seq_len: 
                input_ids += [tokenizer.pad_token_id] 
    else:
        input_ids = input_ids[:max_seq_len - 1] + [   
            tokenizer.eos_token_id]


    model.model.encoder.config.gradient_checkpointing = True
    model.model.decoder.config.gradient_checkpointing = True

    res_ids = model.generate(torch.tensor([input_ids]),
                                        max_length=max_len,
                                        num_beams=5,
                                        no_repeat_ngram_size = 3,
                                        eos_token_id=tokenizer.eos_token_id,
                                        bad_words_ids=[[tokenizer.unk_token_id]])        
    res = tokenizer.batch_decode(res_ids.tolist(), skip_special_tokens=True)[0]
    
    return res

# define function to compute ROGUE score
import numpy as np
from rouge_score import rouge_scorer

filename = 0

def compute_rouge(row):
    global filename

    # generate summary
    max_seq_len = 9216
    predictions = summarize(row['text'], max_seq_len)
    labels = row['abstract']

    print(f'predictions: {predictions}')
    print(f'labels: {labels}')

    # 
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
    scores = scorer.score(labels, predictions)

    print(scores)

    row['rouge1'] = scores["rouge1"].fmeasure
    row['rouge2'] = scores["rouge2"].fmeasure
    row['rougeL'] = scores["rougeL"].fmeasure
    row['rougeLsum'] = scores["rougeLsum"].fmeasure
    
    row['rouge1precision'] = scores["rouge1"].precision
    row['rouge2precision'] = scores["rouge2"].precision
    row['rougeLprecision'] = scores["rougeL"].precision
    row['rougeLsumprecision'] = scores["rougeLsum"].precision

    row['rouge1recall'] = scores["rouge1"].recall
    row['rouge2recall'] = scores["rouge2"].recall
    row['rougeLrecall'] = scores["rougeL"].recall
    row['rougeLsumrecall'] = scores["rougeLsum"].recall

    # Data to be written
    dictionary = {
        "label": labels,
        "prediction": predictions,
        "rogue1": row['rouge1'],
        "rogue2": row['rouge2'],
        "rogueL": row['rougeL'],
        "rogueLsum": row['rougeLsum'],
        "rogue1precision": row['rouge1precision'],
        "rogue2precision": row['rouge2precision'],
        "rogueLprecision": row['rougeLprecision'],
        "rogueLsumprecision": row['rougeLsumprecision'],
        "rogue1recall": row['rouge1recall'],
        "rogue2recall": row['rouge2recall'],
        "rogueLrecall": row['rougeLrecall'],
        "rogueLsumrecall": row['rougeLsumrecall'],
    }


    # Serializing json
    json_object = json.dumps(dictionary, indent=4)
    
    # Writing to sample.json
    with open(f"./predictions/primer-{filename}.json", "w") as outfile:
        outfile.write(json_object)

    filename += 1

    # Extract the median scores
    return row

from datasets import Dataset

eval_dataset = Dataset.from_dict({"text": test_texts, "abstract": test_abstracts})

data_stats = eval_dataset.map(compute_rouge)

# compute average scores
rouge1 = sum(data_stats['rouge1']) / len(data_stats['rouge1'])
rouge2 = sum(data_stats['rouge2']) / len(data_stats['rouge2'])
rougeL = sum(data_stats['rougeL']) / len(data_stats['rougeL'])
rougeLsum = sum(data_stats['rougeLsum']) / len(data_stats['rougeLsum'])

rouge1precision = sum(data_stats['rouge1precision']) / len(data_stats['rouge1precision'])
rouge2precision = sum(data_stats['rouge2precision']) / len(data_stats['rouge2precision'])
rougeLprecision = sum(data_stats['rougeLprecision']) / len(data_stats['rougeLprecision'])
rougeLsumprecision = sum(data_stats['rougeLsumprecision']) / len(data_stats['rougeLsumprecision'])

rouge1recall = sum(data_stats['rouge1recall']) / len(data_stats['rouge1recall'])
rouge2recall = sum(data_stats['rouge2recall']) / len(data_stats['rouge2recall'])
rougeLrecall = sum(data_stats['rougeLrecall']) / len(data_stats['rougeLrecall'])
rougeLsumrecall = sum(data_stats['rougeLsumrecall']) / len(data_stats['rougeLsumrecall'])

# sanity check (should be equal to the number of examples)
print(f"len(data_stats['rouge1']): {len(data_stats['rouge1'])}")

print(f'average rouge1: {rouge1}')
print(f'average rouge2: {rouge2}')
print(f'average rougeL: {rougeL}')
print(f'average rougeLsum: {rougeLsum}')

print(f'average rouge1 precision: {rouge1precision}')
print(f'average rouge2 precision: {rouge2precision}')
print(f'average rougeL precision: {rougeLprecision}')
print(f'average rougeLsum precision: {rougeLsumprecision}')

print(f'average rouge1 recall: {rouge1recall}')
print(f'average rouge2 recall: {rouge2recall}')
print(f'average rougeL recall: {rougeLrecall}')
print(f'average rougeLsum recall: {rougeLsumrecall}')