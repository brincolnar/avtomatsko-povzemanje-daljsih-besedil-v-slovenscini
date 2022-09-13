import logging
import os
import math
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForMaskedLM, RobertaForMaskedLM, CamembertForMaskedLM, RobertaTokenizerFast, TextDataset, DataCollatorForLanguageModeling, Trainer
from transformers import TrainingArguments, HfArgumentParser
from transformers  import LongformerSelfAttention
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class SlobertaLongSelfAttention(LongformerSelfAttention):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value = None,
        output_attentions=False,
    ):
        attention_mask = attention_mask.squeeze(dim=2).squeeze(dim=1)
        is_index_masked = attention_mask < 0
        is_index_global_attn = attention_mask > 0
        is_global_attn = any(is_index_global_attn.flatten())


        return super().forward(hidden_states, 
                               is_index_masked=is_index_masked, 
                               is_index_global_attn=is_index_global_attn, 
                               is_global_attn=is_global_attn,
                               attention_mask=attention_mask, 
                               output_attentions=output_attentions)

class SlobertaLongForMaskedLM(CamembertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.roberta.encoder.layer):
            # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
            layer.attention.self = SlobertaLongSelfAttention(config, layer_id=i)


import torch
import copy

def create_long_model(save_model_to, attention_window, max_pos):
    model = CamembertForMaskedLM.from_pretrained("EMBEDDIA/sloberta")    
    
    
    tokenizer = AutoTokenizer.from_pretrained("EMBEDDIA/sloberta",model_max_length=max_pos)
    
    config = model.config

    # extend position embeddings
    tokenizer.model_max_length = max_pos
    
    # pad to max sequence
    tokenizer.padding = 'max_length'
    tokenizer.init_kwargs['model_max_length'] = max_pos
    current_max_pos, embed_size = model.roberta.embeddings.position_embeddings.weight.shape
    max_pos += 2  # NOTE: RoBERTa has positions 0,1 reserved, so embedding size is max position + 2
    config.max_position_embeddings = max_pos
    assert max_pos > current_max_pos
    # allocate a larger position embedding matrix
    new_pos_embed = model.roberta.embeddings.position_embeddings.weight.new_empty(max_pos, embed_size)
    # copy position embeddings over and over to initialize the new position embeddings
    k = 2
    step = current_max_pos - 2
    while k < max_pos - 1:
        new_pos_embed[k:(k + step)] = model.roberta.embeddings.position_embeddings.weight[2:]
        k += step
    
    model.roberta.embeddings.position_embeddings.weight.data = new_pos_embed
    
    # MISSING in original notebook
    model.roberta.embeddings.position_ids.data = torch.tensor([i for i in range(max_pos)]).reshape(1, max_pos)

    # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
    config.attention_window = [attention_window] * config.num_hidden_layers
    for i, layer in enumerate(model.roberta.encoder.layer):
        longformer_self_attn = LongformerSelfAttention(config, layer_id=i)
        longformer_self_attn.query = layer.attention.self.query
        longformer_self_attn.key = layer.attention.self.key
        longformer_self_attn.value = layer.attention.self.value

        longformer_self_attn.query_global = copy.deepcopy(layer.attention.self.query)
        longformer_self_attn.key_global = copy.deepcopy(layer.attention.self.key)
        longformer_self_attn.value_global = copy.deepcopy(layer.attention.self.value)

        layer.attention.self = longformer_self_attn
    
    logger.info(f'saving model to {save_model_to}')
    model.save_pretrained(save_model_to)
    tokenizer.save_pretrained(save_model_to)
    return model, tokenizer   

def copy_proj_layers(model):
    for i, layer in enumerate(model.roberta.encoder.layer):
        layer.attention.self.query_global = layer.attention.self.query
        layer.attention.self.key_global = layer.attention.self.key
        layer.attention.self.value_global = layer.attention.self.value
    return model

def pretrain_and_evaluate(args, model, tokenizer, eval_only, model_path, block_size):
    val_dataset = TextDataset(tokenizer=tokenizer,
                              file_path=args.val_datapath,
                              block_size=block_size)# originally tokenizer.model_max_len)
    print(val_dataset)

    if eval_only:
        train_dataset = val_dataset
    else:
        logger.info(f'Loading and tokenizing training data is usually slow: {args.train_datapath}')
        train_dataset = TextDataset(tokenizer=tokenizer,
                                    file_path=args.train_datapath,
                                    block_size=block_size)# originally tokenizer.model_max_len)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    trainer = Trainer(model=model, args=args, data_collator=data_collator,
                      train_dataset=train_dataset, eval_dataset=val_dataset)

    eval_loss = trainer.evaluate()
    print(f'eval_loss: {eval_loss}')
    eval_loss = eval_loss['eval_loss']
    logger.info(f'Initial eval bpc: {eval_loss/math.log(2)}')
    
    if not eval_only:
        trainer.train(model_path=model_path)
        trainer.save_model()

        eval_loss = trainer.evaluate()
        eval_loss = eval_loss['eval_loss']
        logger.info(f'Eval bpc after pretraining: {eval_loss/math.log(2)}')

@dataclass
class ModelArgs:
    attention_window: int = field(default=512, metadata={"help": "Size of attention window"})
    max_pos: int = field(default=20480, metadata={"help": "Maximum position"})

parser = HfArgumentParser((TrainingArguments, ModelArgs,))


training_args, model_args = parser.parse_args_into_dataclasses(look_for_args_file=False, args=[
    #'script.py',
    '--output_dir', 'tmp',
    '--warmup_steps', '500',
    '--learning_rate', '0.00003',
    '--weight_decay', '0.01',
    '--adam_epsilon', '1e-6',
    '--max_steps', '3000',
    '--logging_steps', '500',
    '--save_steps', '500',
    '--max_grad_norm', '5.0',
    '--per_gpu_eval_batch_size', '8',
    '--per_gpu_train_batch_size', '2',  # 32GB gpu with fp32
    # '--device', 'cuda0',  # one GPU
    '--gradient_accumulation_steps', '32',
    # '--evaluate_during_training',
    '--do_train',
    '--do_eval',
])
print(model_args)

training_args.train_datapath = '../podatki/Macocu/train.txt'
training_args.val_datapath = '../podatki/Macocu/test.txt'
# training_args.prediction_loss_only = True

# important, otherwise we run out of memory on CUDA
training_args.auto_find_batch_size  = True
training_args.save_strategy = 'epoch'

model_path = f'{training_args.output_dir}/sloberta-{model_args.max_pos}'
if not os.path.exists(model_path):
    os.makedirs(model_path)

logger.info(f'Converting sloberta into sloberta-{model_args.max_pos}')
model, tokenizer = create_long_model(
    save_model_to=model_path, attention_window=model_args.attention_window, max_pos=20480)

logger.info(f'Loading the model from {model_path}')

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = SlobertaLongForMaskedLM.from_pretrained(model_path)

logger.info(f'Pretraining sloberta-{model_args.max_pos} ... ')

training_args.max_steps = 50 # 3
pretrain_and_evaluate(training_args, model, tokenizer, eval_only=False, model_path=model_path, block_size=model_args.max_pos)
