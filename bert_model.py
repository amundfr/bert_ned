# source: [2]
import logging
import csv
import numpy as np
import pandas as pd

from bert import tokenization
import tensorflow as tf
import tensorflow_hub as hub

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

logging.basicConfig(level=logging.INFO)
tf.enable_eager_execution()

class BertModel:
    def __init__(self):
        # Load the BERT model as a KerasLayer from TensorFlow Hub
        module_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
        self.bert_layer = hub.KerasLayer(module_url, trainable=True)
        self.tokenizer = None
        self.model = None
        self.max_len = None

    def get_tokenizer(self):
        if not self.tokenizer:
            # Get the vocabulary and settings for the model
            vocab_file = self.bert_layer.resolved_object.vocab_file.asset_path.numpy()
            do_lower_case = self.bert_layer.resolved_object.do_lower_case.numpy()
            # Initiate the tokenizer from the bert module
            self.tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
        return self.tokenizer

    def get_vocabulary(self, as_type = ''):
        """
        returns as Pandas DataFrame if as_type == 'df'
        """
        tokenizer = self.get_tokenizer()
        if as_type == "df":
            return pd.DataFrame(tokenizer.vocab.keys(), columns=["token"]).reset_index(drop=True)
        else:
            return tokenizer.vocab

    def bert_encode(self, texts, max_len=512):
        """
        Parameters:
            texts: a list of text sequences to be tokenized
            max_len: max sequence length. Longer sequences will be split
        Returns:
            a tripplet of np arrays of tokens, masks and segments
        """
        tokenizer = self.get_tokenizer()

        all_tokens = np.empty((len(texts), max_len), dtype=np.int32)
        all_masks = np.empty((len(texts), max_len), dtype=np.int32)
        all_segments = np.empty((len(texts), max_len), dtype=np.int32)

        for i, text in enumerate(texts):
            text = tokenizer.tokenize(text)
            text = text[:max_len-2]
            # text = [text] # because this is already a token in this example
            input_sequence = ["[CLS]"] + text + ["[SEP]"]
            pad_len = max_len - len(input_sequence)
            tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
            pad_masks = [1] * len(input_sequence) + [0] * pad_len
            segment_ids = [0] * max_len

            all_tokens[i,:] = tokens
            all_masks[i,:] = pad_masks
            all_segments[i,:] = segment_ids

        return all_tokens, all_masks, all_segments

    def build_model(self, max_len=512):
        """
        Parameters:
            max_len: max sequence length
        Returns:
            a TensorFlow Keras model
        """
        self.max_len = max_len
        # Placeholders for the input requirements of the bert layer
        input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
        input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
        segment_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

        # The bert layer provides two output options
        pooled_output, sequence_output = self.bert_layer([input_word_ids, input_mask, segment_ids])
        clf_output = sequence_output[:, 0, :]
        output_dim = sequence_output.shape[-1]
        # print(f"pooled_output.shape: {pooled_output.shape}")
        # print(f"Sequence_output.shape: {sequence_output.shape}")
        # print(f"clf_output.shape: {clf_output.shape}")
        out = tf.keras.layers.Dense(output_dim, activation='linear')(clf_output)
        # net = tf.keras.layers.Dense(64, activation='relu')(clf_output)
        # net = tf.keras.layers.Dropout(0.2)(net)
        # net = tf.keras.layers.Dense(32, activation='relu')(net)
        # net = tf.keras.layers.Dropout(0.2)(net)
        # out = tf.keras.layers.Dense(5, activation='softmax')(net)

        model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
        model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model
        return self.model

    def get_model(self):
        if not self.model:
            print("No model. Run build_model first.")
            raise AttributeError
        return self.model

    def get_embedding(self, texts):
        """
        texts should be list of strings
        """
        tokenizer = self.get_tokenizer()
        model = self.get_model()

        input = self.bert_encode(texts, self.max_len)
        return model.predict(input)

    def word_in_context_embedding(self, text, words):
        """
        Parameters:
            text: a text seqeuence that provides context to the word
            word: a list of word that shows up in the text
        Returns:
            a list of embeddings of shape (len(words), 768)
        """
        raise NotImplementedError
        # Find tokens of sentence

        # Find tokens(s) of each word in sentence

        # Forward pass sentence

        # Find hidden representation of each word's tokens for word embedding

        # Embeddig of each word is the sum or average (?) of its token embeddings



    def get_closest_words_in_vocabulary(self, texts, n_closest=5):
        """
        finds the n_closest closest words to the input texts
        using the cosine distance
        Note that this uses the max_len of the model set before

        Parameters:
            texts: a list of strings that will be embedded
            n_closest: the number of closest words to be returned for each text
        Returns:
            a triplet of
            a list of the n closest words in the embedding to the input texts
            the embedding of these words
            and the embedding of the texts
        """
        return NotImplementedError
        # texts_embeddings = []
        # closest_words = []
        # closest_embeddings = []
        #
        # text_embeddings = self.get_embedding(texts)
        # vocab = self.get_vocabulary("df").tokens[1997:29613]
        # vocab_embedding = get_embedding(vocab)
        # print(vocab_embedding.shape)
        #
        # for text, embedding in zip(texts, texts_embeddings):
        #     print(embedding.shape)
        #     for i in range(n_closest):
        #         simis = cosine_similarity(embedding.reshape(1,-1), vocab_embedding) # of shape len(texts), len(vocab)
        #
        # return closest_words, closest_embeddings, texts_embeddings

if __name__ == '__main__':
    bert_layer = get_bert_layer()
    tokenizer = get_tokenizer(bert_layer)
    # Extract the vocabulary tokens as a DataFrame for later use
    token_vocab = pd.DataFrame(tokenizer.vocab.keys(), columns=["token"]).reset_index(drop=True)

    # Run the tokens through the network to find their embedding
    max_len = 4
    network_input = bert_encode(token_vocab.token, tokenizer, max_len=max_len)
    model = build_model(bert_layer, max_len=max_len)
    print(model.summary())
    test_pred = model.predict(network_input)
    # import time
    # t0 = time.time()
    np.savetxt("embeddings.dat", test_pred, delimiter='\t')
    # t1 = time.time()
    # print(f'Saving the embeddings took {t1-t0} seconds')

    print(model.get_embedding(["cat"]))
