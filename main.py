import sys
import transformers
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# https://github.com/pytorch/pytorch/issues/30966
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

args = sys.argv
if len(args) > 2:
  print("Too many args. Correct usage: python main.py /path/for/output")
  sys.exit(1)
if len(args) == 2:
  output_path = args[1]
else:
  output_path = 'runs/bert_embeddings'
writer = SummaryWriter(output_path)

model = transformers.BertModel.from_pretrained('bert-base-uncased')

tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
words = tokenizer.vocab.keys()

word_embedding = model.embeddings.word_embeddings.weight
writer.add_embedding(word_embedding,
                     metadata  = words,
                     tag = 'word embedding')

position_embedding = model.embeddings.position_embeddings.weight
writer.add_embedding(position_embedding,
                     metadata  = np.arange(position_embedding.shape[0]),
                     tag = 'position embedding')

token_type_embedding = model.embeddings.token_type_embeddings.weight
writer.add_embedding(token_type_embedding,
                     metadata  = np.arange(token_type_embedding.shape[0]),
                     tag = 'tokentype embeddings')

writer.close()
