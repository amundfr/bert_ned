import logging
import itertools
import transformers
import numpy as np
import torch

from sys import version
from torch.utils.tensorboard import SummaryWriter

import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

# embedding1 = torch.nn.Embedding(5, 50)
# embedding2 = torch.nn.Embedding(5, 50)

# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)
# logger.info(f'Python version: {version}')

writer = SummaryWriter('runs/bert_embeddings')
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
