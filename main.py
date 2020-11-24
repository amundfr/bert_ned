import transformers
import numpy as np
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/testing_tensorboard_pt')
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

