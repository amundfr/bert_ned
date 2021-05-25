"""
Author: Amund Faller Råheim

This class wraps a dataset of annotated CoNLL documents.
It extracts mentions tagged with Wikidata IDs
for use as a training corpus.

Requires:
 * a Spacy KnowledgeBase to yield candidates, and
 * a vocabulary for the Spacy KnowledgeBase
"""

from typing import Dict, List
import json
import os

from spacy.vocab import Vocab
from spacy.kb import KnowledgeBase

from lib.wel_minimal.conll_benchmark import ConllDocument, conll_documents


class ConllCandidatesGenerator:
    def __init__(
                self,
                spacy_nlp_vocab_dir: str = "data/vocab",
                spacy_kb_file: str = "data/kb"
            ):
        """
        :param spacy_nlp_vocab_dir: path to directory with spaCy vocab files
        :param spacy_kb_file: path to file with spaCy KnowledgeBase
        """
        # self.spacy_nlp_str = spacy_nlp_str
        self.spacy_nlp_vocab_dir = spacy_nlp_vocab_dir
        self.spacy_kb_file = spacy_kb_file

        # Initialized in get_kb()
        self.kb = None

        self.docs = []
        self.docs_entities = []

    def get_docs(self, file: str = 'conll-wikidata-iob-annotations'):
        """
        :param file: path to file with Wikidata-annotated CoNLL dataset
        :returns: self.docs, reading it from file if not loaded
        """
        if not self.docs:
            if not os.path.isfile(file):
                raise FileNotFoundError(
                        f"Could not find annotated CoNLL file {file}."
                    )

            self.docs = list(conll_documents(file))
        return self.docs

    def del_kb(self):
        """
        Frees up memory by deleting self.kb
        """
        self.kb = None

    def get_kb(self):
        """
        :returns: self.kb, reading it from file if not loaded
        """
        if not self.kb:
            print("Loading vocabulary...")
            vocab = Vocab().from_disk(self.spacy_nlp_vocab_dir)

            print("Loading KB...")
            self.kb = KnowledgeBase(vocab=vocab)
            self.kb.load_bulk(self.spacy_kb_file)
            print("KB loaded!")
        return self.kb

    def write_entities_info(self, file: str = "docs_entities_info.json"):
        """
        Writes self.docs_entities to file.
        File then contains all necessary candidate info,
         which allows candidates to be read from file
         with read_entities_info later
        :param file: file destination of output file
        """
        if not self.docs_entities:
            raise ValueError("ERROR: No candidates to write to file. "
                  "Try the function 'get_candidates' first.")

        print(f"Writing json to file {file} ...")
        with open(file, 'w') as of:
            json.dump(self.docs_entities, of)

    def read_entities_info(self, file: str = "docs_entities_info.json"):
        """
        Read self.docs_entities from file, and returns self.docs_entities
        File should be result of function write_entities_info,
         and gives all necessary candidate info
        :param file: path to file written by write_entities_info
        :returns: self.docs_entities
        """
        if not os.path.isfile(file):
            raise FileNotFoundError(f"Could not find file {file}. "
                  "Try the function write_entities_info first.")

        print("Reading from file...")
        with open(file, 'r') as inf:
            self.docs_entities = json.load(inf)
        return self.docs_entities

    def generate_candidates_for_doc(self, doc: ConllDocument) -> List[Dict]:
        """
        Takes a ConllDocument object with tagged tokens
        (e.g. from conll_documents()).

        Outputs a list of dictionaries for each tagged named entity.
        Each dict has a dict of:
            the ground truth of the entity (as a 'Q-ID' from WikiData),
            the token position of the entity as a tuple (start, end),
            and a list of candidates, represented by their wikidata 'Q-ID'.

        :param doc: a ConllDocument object with tokens tagged with WikiData IDs
        :returns: a list over the tagged named entities, each a dictionary of
                  ground truth, entity position, and candidates
        """
        self.get_kb()
        # The return variable. Stores the list of entities.
        entities = []

        # Inner function to append a label_dict to the entities list
        def add_entity(entity_span_s, entity_span_e, entity_tokens, entity_gt):
            entity_text = ' '.join(entity_tokens)
            entity_candidates = [
                    c.entity_ for c in self.kb.get_candidates(entity_text)
                ]
            entity_span = [entity_span_s, entity_span_e]

            entities.append(
                    {'Position': entity_span,
                     'GroundTruth': entity_gt,
                     'Candidates': entity_candidates}
                )

        # Helper variables for the iteration:
        # Tokens belonging to current entity
        collected_tokens = []
        # Tag of the current entity (the ground truth)
        current_entity_tag = None
        # Position of the first entity token in the document tokens list
        span_start = None

        # Enumerate the document's list of tokens
        for i_token, token in enumerate(doc.tokens):

            # If we are looking at the beginning of a named entity
            if token.true_label.startswith("Q") or token.true_label == "B":

                # Check if we already have collected a named entity
                # This is the case when two named entities follow each other
                if len(collected_tokens) > 0:
                    add_entity(span_start, i_token-1,
                               collected_tokens, current_entity_tag)

                span_start = i_token
                collected_tokens = [token.text]
                current_entity_tag = token.true_label

            # If we are looking at the continuation of a named entity
            elif token.true_label == 'I':
                collected_tokens.append(token.text)

            # If we're not looking at a token in a named entity
            else:
                # If we have passed the end of a named entity
                if len(collected_tokens) > 0:
                    add_entity(span_start, i_token-1,
                               collected_tokens, current_entity_tag)

                collected_tokens = []

        # If the last tokens were a named entity
        if len(collected_tokens) > 0:
            add_entity(span_start, len(doc.tokens)-1,
                       collected_tokens, current_entity_tag)

        return entities

    def get_docs_entities(
                self,
                f: str = None,
                del_kb: bool = True
            ) -> List[List[Dict]]:
        """
        Iterates CoNLL documents and gets the cadidates for all mentions
        :param f: file with tagged conll documents
        :param del_kb: Whether to delete the KB object to free up space
        :returns: a list of dicts with lists of info about entities
        """

        # Generate if not cached
        if not self.docs_entities:

            if self.docs:
                self.docs = []

            for conll_doc in self.get_docs(f):
                self.docs_entities.append(
                        self.generate_candidates_for_doc(conll_doc)
                    )

            if del_kb:
                print("Deleting Spacy KB object...")
                self.del_kb()

        return self.docs_entities

    def print_candidate_stats(self):
        """
        Prints metrics about generated candidates
        """
        if not self.docs_entities:
            print("No candidates info.")
            return

        # Number of entities with no candidates (no data points)
        n_no_cand = 0
        # Number of entities where ground truth is among the candidates
        n_pos_labels = 0
        # Number of entities where GT is not among the candidates
        n_no_pos_labels = 0
        # Number of candidates excluding the GT candidate
        n_neg_labels = 0

        # Total number of named entities
        n_ne = 0
        # Only named entities in the wikidata KB
        n_ne_in_kb = 0
        # Number of named entities not linked to Wikidata KB
        n_ne_bs = 0
        # Number of candidates that belong to entities with no GT
        n_b_cands = 0

        for doc_entities in self.docs_entities:
            for entity in doc_entities:
                n_ne += 1

                if len(entity['Candidates']) == 0:
                    n_no_cand += 1
                elif entity['GroundTruth'] in entity['Candidates']:
                    n_pos_labels += 1
                    n_neg_labels += len(entity['Candidates']) - 1
                else:
                    n_no_pos_labels += 1
                    n_neg_labels += len(entity['Candidates'])

                if entity['GroundTruth'] == 'B':
                    n_ne_bs += 1
                    n_b_cands += len(entity['Candidates'])
                else:
                    n_ne_in_kb += len(entity['Candidates'])

        n_cand = n_pos_labels + n_neg_labels

        print(f"{n_ne: >7,} named entities in total")
        print(f"{n_cand: >7,} candidates in total "
              f"(total number of data points)")
        print(f"{n_pos_labels: >7,} / {n_cand: >7,} positive labels "
              f"({100 * n_pos_labels / n_cand: >5.2f} % all all labels )")
        print(f"{n_neg_labels: >7,} / {n_cand: >7,} negative labels "
              f"({100 * n_neg_labels / n_cand: >5.2f} % all all labels )")

        print(f"{n_no_cand: >7,} / {n_ne: >7,} "
              f"named entities have no candidates")
        print(f"{n_no_pos_labels: >7,} / {n_ne: >7,} "
              f"named entities where correct label is not among candidates")
        print(f"{n_ne_in_kb: >7,} / {n_cand: >7,} "
              f"candidates tagged with GT in Wikidata KB")
        print(f"{n_ne_bs: >7,} / {n_cand: >7,} "
              f"candidates for named entities not in Wikidata KB")

        print(f"{n_cand/n_ne:.1f} average number of candidates per entity")
