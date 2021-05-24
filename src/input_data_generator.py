"""
Author: Amund Faller Råheim

This class generates tokenized input data for BERT using Wikipedia abstracts.
Uses document, mention and candidate information from ConllCandidatesGenerator.

Requires:
a tsv file with Wikidata ID, Wikipedia article title, and Wikipedia abstract
    used as context for candidate entities
a file with CoNLL documents, used as contextual data for the mention
"""

import time
from os.path import isfile
from math import floor
from typing import List

from torch import BoolTensor, ShortTensor, cat
from transformers import BertTokenizerFast

from lib.wel_minimal.conll_benchmark import ConllDocument


class InputDataGenerator:
    def __init__(self,
                 wikipedia_abstracts_file: str = 'wikidata-wikipedia.tsv',
                 tokenizer_pretrained_id: str = 'bert-base-uncased'):
        self.wikipedia_abstracts_file = wikipedia_abstracts_file
        self.wikipedia_abstracts = {}

        uncased = tokenizer_pretrained_id.endswith('uncased')
        self.tokenizer = BertTokenizerFast.from_pretrained(
                tokenizer_pretrained_id,
                do_lower_case=uncased
            )

        # Token IDS for three special tokens
        self.cls_id = self.tokenizer.convert_tokens_to_ids('[CLS]')
        self.sep_id = self.tokenizer.convert_tokens_to_ids('[SEP]')
        self.pad_id = self.tokenizer.convert_tokens_to_ids('[PAD]')

    def read_wikipedia_abstracts(self, f: str = ''):
        if not self.wikipedia_abstracts_file and not f:
            print("Must provide a wikipedia abstracts file either when initializing"
                  " or when calling read_wikipedia_abstracts.")
            return 0
        elif not isfile(f) and not isfile(self.wikipedia_abstracts_file):
            print("Can't find wikipedia abstracts file.")
            return 0
        elif isfile(f):
            self.wikipedia_abstracts_file = f

        print("Reading Wikipedia Abstracts File...", end='')
        for line in open(self.wikipedia_abstracts_file, 'r'):
            values = line[:-1].split('\t')
            self.wikipedia_abstracts[values[0]] = (values[1], values[2].strip())
        print("Done!")

        return 1

    def get_wikipedia_abstracts(self):
        if self.wikipedia_abstracts == {}:
            success = self.read_wikipedia_abstracts()
            if not success:
                print("ERROR: No Wikipedia Abstracts.")
                return
        return self.wikipedia_abstracts

    def get_vectorized_data(self,
                            document: ConllDocument,
                            doc_entities: List,
                            max_len: int = 512):
        """
        Function yields the three input vectors to BERT
        for a given CoNLL document with a doc_entities list,
        its binary label ("are they the same?"),
        and the position in the tokens
        I.e. for each named entity, a different data point
        is yielded for each candidate

        :param document: a ConllDocument object
        :param doc_entities: a list of dicts
        :param max_len: maximum length of a returned vector
        :returns: Tuple of: Three tensors with input_ids, attention_mask, 
                    token_type_ids, and a boolean label
        """
        input_ids = []
        attention_mask = []
        token_type_ids = []
        label = None

        # This takes a while, and up to 5 GB of RAM
        wikipedia_abstracts = self.get_wikipedia_abstracts()

        # Tokenize the document first
        doc_words = [t.text for t in document.tokens]
        doc_tokenized = self.tokenizer.encode(
                doc_words,
                truncation=False,
                add_special_tokens=False,
                is_split_into_words=True,
                verbose=False
            )

        for entity_dict in doc_entities:
            entity_span = entity_dict['Position']
            gt = entity_dict['GroundTruth']

            # Filters out data with no label
            if gt == 'B':
                continue

            for candidate in entity_dict['Candidates']:

                # If there is no wikipedia abstract, skip candidate
                if candidate not in wikipedia_abstracts:
                    continue

                label = True if candidate == gt else False

                # Max number of tokens for the abstract context.
                # Half of the total length, excluding two [SEP] and one [CLS] tokens
                max_cand_len = floor((max_len - 3) / 2)

                # Start with the candidate title
                cand_context = wikipedia_abstracts[candidate][0]
                # Add a truncated part of the abstract (faster for very long abstracts)
                cand_context += ' ' + ' '.join(wikipedia_abstracts[candidate][1].split(' ')[:int(max_cand_len*1.5)])
                # Tokenize with truncation
                cand_tokens = self.tokenizer.encode(
                                    cand_context,
                                    max_length=max_cand_len,
                                    truncation=True,
                                    add_special_tokens=False
                                )

                # The quota for the named entity context in number of tokens
                entity_text = doc_words[entity_span[0]:entity_span[1]+1]
                ne_tokenized = self.tokenizer.encode(
                        entity_text,
                        add_special_tokens=False,
                        is_split_into_words=True
                    )
                max_context_len = max_len - 3 - len(cand_tokens) - len(ne_tokenized)

                # Make sure we get at least some words after the token
                start = max(0, entity_span[1] + 15 - max_context_len)
                end = start + max_context_len
                entity_tokens = ne_tokenized + doc_tokenized[start:end]

                input_ids = [self.cls_id] + entity_tokens + [self.sep_id] + cand_tokens + [self.sep_id]

                pad_len = max_len - len(input_ids)

                attention_mask = [1] * len(input_ids) + [0] * pad_len

                input_ids += [self.pad_id] * pad_len

                token_type_ids = [0] * (2 + len(entity_tokens)) + [1] * (1 + len(cand_tokens)) + [0] * pad_len

                # Save in minimal format. Will need to be recast to torch.long for BERT
                input_ids = ShortTensor(input_ids).unsqueeze(0)
                attention_mask = BoolTensor(attention_mask).unsqueeze(0)
                token_type_ids = BoolTensor(token_type_ids).unsqueeze(0)

                yield input_ids, attention_mask, token_type_ids, label

    def generate_for_conll_data(self,
                                docs: List,
                                docs_entities: List,
                                max_len: int = 512,
                                progress: bool = False):
        """
        Generates tokenized BERT input vectors for input documents and entity info

        :param docs: a list of ConllDocuments, obtained
            from ConllCandidatesGenerator's
        :param docs_entities: a list with entity info, obtained
            from ConllCandidatesGenerator's get_docs_entities()
        :param max_len: maximum length of tokenized input
        :param progress: print progress
        """
        data = []
        t0 = time.time()
        n = len(docs)

        for i_doc, doc_tuple in enumerate(zip(docs, docs_entities)):
            if progress and (not i_doc == 0) and (i_doc % (n / 20) <= 1):
                s_left = (n - i_doc) * (time.time() - t0) / i_doc
                print(f"Progress: {i_doc:>6} / {n:} ({i_doc / n * 100: >4.1f} %)" +
                      f" time left: {s_left / 3600:.0f}:{(s_left % 3600) / 60:02.0f}:{s_left % 60:02.0f} hh:mm:ss", end='\r')

            for vectors in self.get_vectorized_data(doc_tuple[0], doc_tuple[1], max_len):
                assert len(vectors) == 4
                data.append(vectors)

        # Save in minimal format. Will need to be recast to torch.long for BERT
        input_ids = cat([d[0] for d in data])
        attention_mask = cat([d[1] for d in data])
        token_type_ids = cat([d[2] for d in data])
        labels = BoolTensor([d[3] for d in data])

        return input_ids, attention_mask, token_type_ids, labels
