"""
Author: Amund Faller Råheim

This class makes dataloaders and splits the dataset.
Takes the output of InputDataGenerator as input.
"""

from typing import List, Tuple
from random import sample
from os.path import isdir, isfile, join
import json

from torch.utils.data import TensorDataset, Subset, \
        DataLoader, RandomSampler, SequentialSampler
from torch import Tensor, load, save, cat


class DatasetGenerator:
    def __init__(self,
                 input_ids: Tensor = Tensor(),
                 attention_mask: Tensor = Tensor(),
                 token_type_ids: Tensor = Tensor(),
                 labels: Tensor = Tensor()):
        # The data tensors
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.labels = labels
        # Place holders for TensorDataset objects
        self.dataset = None
        self.balanced_dataset = None
        # Lists saying which document index a data point
        # in the respective datasets come from
        self.dataset_to_doc = []
        self.balanced_dataset_to_doc = []
        self.dataset_to_entity = []
        self.balanced_dataset_to_entity = []
        self.dataset_to_candidate = []
        self.balanced_dataset_to_candidate = []
        # Subsets of a dataset defined in split_dataset
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        # Default directory and files to save and load tensors
        self.file_dir = 'data/vectors'
        self.file_names = ['data_vectors_input_ids.pt',
                           'data_vectors_attention_mask.pt',
                           'data_vectors_token_type_ids.pt',
                           'data_vectors_labels.pt']

    def read_from_directory(self, vectors_dir: str = 'data/vectors'):
        if not isdir(vectors_dir):
            raise FileNotFoundError(f"Could not find directory at {vectors_dir}.")

        if not all([isfile(join(vectors_dir, f)) for f in self.file_names]):
            raise FileNotFoundError(f"Could not find all files in directory at {vectors_dir}."
                                    " Try function read_vectors_from_file.")

        self.file_dir = vectors_dir
        self.input_ids = load(join(vectors_dir, self.file_names[0]))
        self.attention_mask = load(join(vectors_dir, self.file_names[1]))
        self.token_type_ids = load(join(vectors_dir, self.file_names[2]))
        self.labels = load(join(vectors_dir, self.file_names[3])).unsqueeze(-1)

    def read_from_files(self,
                        f_input_ids: str = 'data_vectors_input_ids.pt',
                        f_attention_mask: str = 'data_vectors_attention_mask.pt',
                        f_token_type_ids: str = 'data_vectors_token_type_ids.pt',
                        f_labels: str = 'data_vectors_labels.pt',
                        directory: str = 'data/vectors'
                        ):
        self.file_names = [f_input_ids, f_attention_mask,
                           f_token_type_ids, f_labels]
        self.file_dir = directory
        self.read_from_directory(self.file_dir)

    def write_to_files(self, vectors_dir: str = 'data/vectors'):
        print(f"Writing vectors to directory {vectors_dir}...")
        save(self.input_ids, join(vectors_dir, self.file_names[0]))
        save(self.attention_mask, join(vectors_dir, self.file_names[1]))
        save(self.token_type_ids, join(vectors_dir, self.file_names[2]))
        save(self.labels, join(vectors_dir, self.file_names[3]))

    def write_balanced_dataset_to_files(self, dataset_dir: str = 'data/balanced_dataset'):
        if not self.balanced_dataset:
            raise ValueError("Balanced dataset not initialized. Try get_balanced_dataset.")

        with open(join(dataset_dir, 'balanced_dataset_to_doc'), 'w') as of:
            json.dump(self.balanced_dataset_to_doc, of)
        with open(join(dataset_dir, 'balanced_dataset_to_entity'), 'w') as of:
            json.dump(self.balanced_dataset_to_entity, of)

        for i, tensor in enumerate(self.balanced_dataset.tensors):
            save(tensor, join(dataset_dir, self.file_names[i]))

    def read_balanced_dataset(self, dataset_dir: str = 'data/balanced_dataset'):
        if not isdir(dataset_dir):
            raise FileNotFoundError(f"Could not find directory at {dataset_dir}.")

        with open(join(dataset_dir, 'balanced_dataset_to_doc')) as f:
            self.balanced_dataset_to_doc = json.load(f)
        with open(join(dataset_dir, 'balanced_dataset_to_entity')) as f:
            self.balanced_dataset_to_entity = json.load(f)

        if not all([isfile(join(dataset_dir, f)) for f in self.file_names]):
            raise FileNotFoundError(f"Could not find all files in directory at {dataset_dir}."
                  " Try function read_vectors_from_file.")

        input_ids = load(join(dataset_dir, self.file_names[0]))
        attention_mask = load(join(dataset_dir, self.file_names[1]))
        token_type_ids = load(join(dataset_dir, self.file_names[2]))
        labels = load(join(dataset_dir, self.file_names[3]))
        self.balanced_dataset = TensorDataset(input_ids, attention_mask, token_type_ids, labels)

    def get_tensor_dataset(self) -> TensorDataset:
        if not self.dataset:
            self.dataset = TensorDataset(self.input_ids, self.attention_mask,
                                         self.token_type_ids, self.labels)
        return self.dataset

    def get_dataset_to_x(self, docs_entities: List = None) -> Tuple[List, List, List]:
        """
        Iterating the docs_entities list of mentions, 
        this function generates the mapping of the dataset (unsplit)
        to doc index, mention index (starting at 0 from each doc),
        and the candidate ID. 
        
        This is also accomplished in get_balanced_dataset,
        but much faster if the balanced dataset is not desired.

        :returns: three lists of the length of the dataset with
            doc indices, in-doc mention indices, and candidate IDs
        """
        if self.dataset_to_doc and self.dataset_to_entity and self.dataset_to_candidate:
            return self.dataset_to_doc, self.dataset_to_entity, self.dataset_to_candidate
        else:
            self.dataset_to_doc = []
            self.dataset_to_entity = []
            self.dataset_to_candidate = []

            for i_doc, doc_entities in enumerate(docs_entities):
                # Iterate Named Entities in current doc
                for i_entity, entity_info in enumerate(doc_entities):
                    # Skip 'B'-entities (entities not in KB)
                    if entity_info['GroundTruth'] != 'B' and entity_info['Candidates']:
                        # All these candidates belong to current doc
                        self.dataset_to_doc.extend([i_doc] * len(entity_info['Candidates']))
                        self.dataset_to_entity.extend([i_entity] * len(entity_info['Candidates']))
                        self.dataset_to_candidate.extend(entity_info['Candidates'])
            return self.dataset_to_doc, self.dataset_to_entity, self.dataset_to_candidate

    def get_balanced_dataset(self, docs_entities: List, n_neg: int = 1) -> TensorDataset:
        """
        Create a smaller, balanced dataset with up to n_neg negative
        sample for each entity.

        :param docs_entities: the docs_entities property from ConllToCandidates
        :param n_neg: the number of negative samples to include for each entity.
            Set to 1 for a more or less balanced dataset
        :returns: A TensorDataset with all positive labels and up to n_neg
            negative labels for each entity
        """
        if self.balanced_dataset:
            return self.balanced_dataset

        full_dataset = self.get_tensor_dataset()

        balanced_tensors = [Tensor().to(dtype=full_dataset[0][0].dtype),
                            Tensor().to(dtype=full_dataset[0][1].dtype),
                            Tensor().to(dtype=full_dataset[0][2].dtype),
                            Tensor().to(dtype=full_dataset[0][3].dtype)]

        # List of doc index for each datapoint in the datasets
        # For the full dataset
        self.dataset_to_doc = []
        # For the balanced_dataset
        self.balanced_dataset_to_doc = []

        # List of entity index for each datapoint in the datasets
        # For the full dataset
        self.dataset_to_entity = []
        # For the balanced_dataset
        self.balanced_dataset_to_entity = []

        # List of candidates for each datapoint in the datasets
        self.dataset_to_candidate = []
        self.balanced_dataset_to_candidate = []

        # Points at indices in the full dataset
        full_dataset_idx = 0
        for i_doc, doc_entities in enumerate(docs_entities):
            # Iterate Named Entities in current doc
            for i_entity, entity_info in enumerate(doc_entities):
                # Skip 'B'-entities (entities not in KB)
                if entity_info['GroundTruth'] != 'B' and entity_info['Candidates']:

                    # All these candidates belong to current doc
                    self.dataset_to_doc.extend([i_doc] * len(entity_info['Candidates']))
                    self.dataset_to_entity.extend([i_entity] * len(entity_info['Candidates']))
                    self.dataset_to_candidate.extend(entity_info['GroundTruth'])

                    # Add any positive datapoint (i.e. where candidate is ground truth)
                    if entity_info['GroundTruth'] in entity_info['Candidates']:
                        local_gt_idx = entity_info['Candidates'].index(entity_info['GroundTruth'])
                        gt_idx = full_dataset_idx + local_gt_idx

                        # Keep track of which document and entity this comes from
                        self.balanced_dataset_to_doc.append(i_doc)
                        self.balanced_dataset_to_entity.append(i_entity)
                        self.balanced_dataset_to_candidate.append(entity_info['GroundTruth'])

                        # Append to all four input vectors
                        for j in range(len(balanced_tensors)):
                            balanced_tensors[j] = cat([
                                balanced_tensors[j],
                                full_dataset[gt_idx][j].view(1, -1)
                            ], dim=0)

                    # Candidates that are not the ground truth
                    neg_cands = [c for c in entity_info['Candidates'] if c != entity_info['GroundTruth']]
                    # Sample all or up to n_neg candidates
                    n_sample = min(n_neg, len(neg_cands))
                    # Sample candidates
                    random_cands = sample(neg_cands, n_sample)

                    for random_cand in random_cands:
                        # Index of the tensors corresponding to this candidate
                        local_cand_idx = entity_info['Candidates'].index(random_cand)
                        cand_idx = full_dataset_idx + local_cand_idx

                        # Keep track of which document and entity this comes from
                        self.balanced_dataset_to_doc.append(i_doc)
                        self.balanced_dataset_to_entity.append(i_entity)
                        self.balanced_dataset_to_candidate.append(random_cand)

                        # Append to all four input vectors
                        for j in range(len(balanced_tensors)):
                            balanced_tensors[j] = cat([
                                    balanced_tensors[j],
                                    full_dataset[cand_idx][j].view(1, -1)
                                    ], dim=0
                                )

                    # Move index pointer to the next entity's data points
                    full_dataset_idx += len(entity_info['Candidates'])

        self.balanced_dataset = TensorDataset(balanced_tensors[0],
                                              balanced_tensors[1],
                                              balanced_tensors[2],
                                              balanced_tensors[3])
        return self.balanced_dataset

    def get_split_dataset(self, split_ratios: List, dataset: str = 'balanced', docs_entities: List = None)\
            -> Tuple[Subset, Subset, Subset]:
        """
        Splits the dataset in given ratios to train, validation and test subsets
        Splits on the documents that the data is from, rather than the datapoints,
        so the exact ratio of the subsets may not be as expected.
        This requires that the functions to initialize the datasets have already been run

        :param split_ratios: the ratios of the train, validation, and split respectively
            as a list of three ratio values
        :param dataset: 'full' or 'balanced' respectively for the full or the balanced dataset
        :param docs_entities: if not 'balanced', this is used to generate dataset_to_doc
        :returns: three torch Subset with training, validation and test data
        """
        if dataset == 'balanced':
            dataset_to_doc = self.balanced_dataset_to_doc
            dataset = self.balanced_dataset
        else:
            dataset_to_doc, _, _ = self.get_dataset_to_x(docs_entities)
            dataset = self.get_tensor_dataset()

        # Number of different docs
        n_docs = dataset_to_doc[-1]

        # ratio of training data, base 1
        train_ratio = split_ratios[0] / sum(split_ratios)
        n_train = int(train_ratio * n_docs)
        # ratio of validation data, base 1
        val_ratio = split_ratios[1] / sum(split_ratios)
        n_val = int(val_ratio * n_docs)
        # the rest is test data
        n_test = n_docs - n_train - n_val

        print(f" Dataset ID ranges:")
        print(f"--      Train: [{1:>4}, {n_train:>4}]")
        print(f"-- Validation: [{n_train+1:>4}, {n_train + n_val:>4}]")
        print(f"--       Test: [{n_train + n_val + 1:>4}, {n_train+n_val+n_test+1:>4}] (Tot: {n_docs+1:>4})\n")

        # Keeping track of the indices in the dataset
        # These are used to define the final Subset
        train_indices = []
        val_indices = []
        test_indices = []

        for i, doc_idx in enumerate(dataset_to_doc):
            if doc_idx < n_train:
                train_indices.append(i)
            elif doc_idx < (n_train + n_val):
                val_indices.append(i)
            else:
                test_indices.append(i)

        print(f"({train_indices[0]: >6}, {train_indices[-1]: >6})   training set slice, "
              f"{(train_indices[-1] - train_indices[0]) / n_train: >6.1f} candidates on average per document")
        print(f"({val_indices[0]: >6}, {val_indices[-1]: >6}) validation set slice, "
              f"{(val_indices[-1] - val_indices[0]) / n_val: >6.1f} candidates on average per document")
        print(f"({test_indices[0]: >6}, {test_indices[-1]: >6})       test set slice, "
              f"{(test_indices[-1] - test_indices[0]) / n_test: >6.1f} candidates on average per document")

        self.train_dataset = Subset(dataset, train_indices)
        self.val_dataset = Subset(dataset, val_indices)
        self.test_dataset = Subset(dataset, test_indices)

        return self.train_dataset, self.val_dataset, self.test_dataset

    def get_data_loaders(self, batch_size: int = 32):
        """
        Using the subsets from get_split_dataset, this makes a DataLoader for each subset

        :param batch_size: batch size for all data loaders
        :returns: a Tuple of three DataLoader
        """
        if not (self.train_dataset and self.val_dataset and self.test_dataset):
            print("The split datasets have not been initialized."
                  "Please try the function get_split_dataset first.")
            return

        # Random sampling of training data
        train_dataloader = DataLoader(
                self.train_dataset,  # The training samples.
                sampler=RandomSampler(self.train_dataset),  # Select batches randomly
                batch_size=batch_size  # Trains with this batch size.
            )

        # Sequential sampling of validation data
        validation_dataloader = DataLoader(
                self.val_dataset,
                sampler=SequentialSampler(self.val_dataset),
                batch_size=batch_size
            )

        # Sequential sampling of test data
        test_dataloader = DataLoader(
                self.test_dataset,
                sampler=SequentialSampler(self.test_dataset),
                batch_size=batch_size
            )

        return train_dataloader, validation_dataloader, test_dataloader

    def print_token_sequence_stats(self):
        # Average length of the left and right sequences
        n_tots = []
        n_left_tokens = []
        n_right_tokens = []
        for att, tt, label in zip(self.attention_mask, self.token_type_ids, self.labels):
            n_tot = len(att[att])
            n_r = len(tt[tt])
            n_l = n_tot - n_r
            n_tots.append(n_tot)
            n_left_tokens.append(n_l)
            n_right_tokens.append(n_r)

        print(f"{sum(n_tots) / len(n_tots) :.1f} average number of tokens")
        print(f"{sum(n_left_tokens) / len(n_left_tokens) :.1f} average tokens in entity sequence.")
        print(f"Min: {min(n_left_tokens)}, max: {max(n_left_tokens)}")
        print(f"{sum(n_right_tokens) / len(n_right_tokens) :.1f} average tokens in candidate sequence.")
        print(f"Min: {min(n_right_tokens)}, max: {max(n_right_tokens)}")

    def get_dataset_balance_info(self) -> Tuple[int, int]:
        """
        :returns: number of negative and positive labels in training dataset
        """
        if not self.train_dataset:
            print("The split datasets have not been initialized."
                  "Please try the function get_split_dataset first.")
            return 0, 0
        # Statistics of the training set
        pos = 0
        neg = 0
        for d in self.train_dataset:
            pos += int(d[3])
            neg += int(not d[3])

        print(f"Negative labels: {neg:,}, positive: {pos:,}, negative/positive: {neg / pos:.3f}")
        return neg, pos
