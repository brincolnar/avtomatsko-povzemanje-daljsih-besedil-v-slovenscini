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
eighty_percent = int(len(texts) * 0.8)

train_texts = texts[:eighty_percent]
train_abstracts = abstracts[:eighty_percent]

test_texts = texts[eighty_percent:]
test_abstracts = abstracts[eighty_percent:]


# evaluation set (= 20% of train set)
from sklearn.model_selection import train_test_split

train_texts, val_texts, train_abstracts, val_abstracts = train_test_split(
    train_texts, train_abstracts, test_size=.2)

print(f'len(val_texts): {len(val_texts)}')
print(f'len(val_abstracts): {len(val_abstracts)}')

print(f'len(train_texts): {len(train_texts)}')
print(f'len(train_abstracts): {len(train_abstracts)}')

# creating HF Dataset objects from raw data
from datasets import Dataset

train_dataset = Dataset.from_dict(
    {"text": train_texts, "abstract": train_abstracts})
val_dataset = Dataset.from_dict({"text": val_texts, "abstract": val_abstracts})

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

tokenizer = MBart50Tokenizer.from_pretrained("./tmp/20k/mbart-long")

model = LongformerEncoderDecoderForConditionalGeneration.from_pretrained(
    "./tmp/20k/mbart-long")
print(model)

# map article and summary len to dict with length information
def map_to_length(x):
    x["text_len"] = len(tokenizer(x["text"]).input_ids)
    x["text_longer_20480"] = int(x["text_len"] > 20480)
    x["abstract_len"] = len(tokenizer(x["abstract"]).input_ids)
    return x


data_stats = train_dataset.map(map_to_length, num_proc=4)

print(f'Text Mean: {sum(data_stats["text_len"]) / len(data_stats["text_len"])}, Abstract Mean: {sum(data_stats["abstract_len"]) / len(data_stats["abstract_len"])}, % > 12 288: {sum(data_stats["text_longer_12288"]) / len(data_stats["text_longer_12288"])}')

tokenizer.cur_lang_code_id = "sl_SI" 
print(tokenizer.cur_lang_code_id)

tokenizer.src_lang = "sl_SI"
print(tokenizer.src_lang)

tokenizer.tgt_lang = "sl_SI"
print(tokenizer.tgt_lang)


max_input_length = 20480

max_target_length = 1024

# tokenizing strings + padding and truncation
def preprocess_function(examples):

    # tokenize whole texts
    model_inputs = tokenizer(
        examples["text"], max_length=max_input_length, padding=True, truncation=True
    )

    # Set up the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        # tokenize abstracts
        labels = tokenizer(
            examples["abstract"], max_length=max_target_length, padding=True, truncation=True
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_tokenized_dataset = train_dataset.map(preprocess_function, batched=True)
val_tokenized_dataset = val_dataset.map(preprocess_function, batched=True)

# remove string columns
train_tokenized_dataset = train_tokenized_dataset.remove_columns(['text', 'abstract'])
val_tokenized_dataset = val_tokenized_dataset.remove_columns(['text', 'abstract'])

train_tokenized_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"])
val_tokenized_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"])

#  This is required to ensure that the decoder only sees the 
#  previous ground truth labels and not the current or future ones, 
#  which would be easy for the model to memorize.
from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Treniranje
from transformers import Seq2SeqTrainingArguments


batch_size = 4 # possibly lower
num_train_epochs = 1

training_args = Seq2SeqTrainingArguments(
    output_dir='./results/20k/mbart-long',          # output directory
    evaluation_strategy="steps",
    eval_steps=1,
    learning_rate=5.6e-5,
    num_train_epochs=num_train_epochs,              # total number of training epochs
    per_device_train_batch_size=batch_size,  # batch size per device during training
    per_device_eval_batch_size=batch_size,   # batch size for evaluation
    weight_decay=0.01,               # strength of weight decay
    save_strategy="steps",
    save_steps=50
    #logging_dir='./logs',            # directory for storing logs
    #logging_steps=logging_steps,
)

from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    model,
    training_args,
    train_dataset=train_tokenized_dataset,
    eval_dataset=val_tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)


trainer.train()