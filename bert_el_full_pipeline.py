from configparser import ConfigParser
import time

from src.conll_candidates_generator import ConllCandidatesGenerator
from src.input_data_generator import InputDataGenerator
from src.dataset_generator import DatasetGenerator
from src.bert_model import BertBinaryClassification, load_bert_from_file, \
    save_bert_to_file
from src.trainer import ModelTrainer
from src.evaluation import plot_training_stats, read_result_and_evaluate

import torch
import numpy as np
import random


# -- Setup --------------------------------------------------------------------


# Set the seed value everywhere to make this reproducible
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


# Wrapper function to time executions
def timer(func):
    def wrapper(*args, **kwargs):
        print(f"----\n  Timing function ...\n----")
        t0 = time.time()
        res = func(*args, **kwargs)
        t1 = time.time()
        d_t_str = time.strftime("%H:%M:%S hh:mm:ss", time.gmtime(t1-t0))
        print(f"----\n  Function ran from {time.ctime(t0)} to {time.ctime(t1)}")
        print(f"  Took: {d_t_str}\n----")
        return res
    return wrapper


config = ConfigParser()
config.read('config.ini')


# -- Generate candidates for mentions in CoNLL dataset ------------------------


@timer
def candidate_generation():
    candidate_generator = ConllCandidatesGenerator(
            spacy_nlp_str=config['DATA']['Spacy NLP Model'],
            spacy_nlp_vocab_dir=config['DATA']['Spacy Vocab Dir'],
            spacy_kb_file=config['DATA']['Spacy KB']
        )
    # candidate_generator.get_entities_and_candidates(config['DATA']['Conll Annotated'])
    candidate_generator.read_entities_info(config['DATA']['Candidate Info'])
    candidate_generator.print_candidate_stats()

    docs_entities = candidate_generator.get_docs_entities()
    docs = candidate_generator.get_docs(config['DATA']['Conll Annotated'])

    del candidate_generator
    return docs_entities, docs


docs_entities, docs = candidate_generation()


# -- Generate input vectors from CoNLL docs and Wikipedia abstracts -----------


@timer
def input_data_generation():
    input_data_generator = InputDataGenerator(
            wikipedia_abstracts_file=config['DATA']['Wikipedia Abstracts'],
            tokenizer_pretrained_id=config['BERT']['Model ID']
        )

    input_vectors = input_data_generator.generate_for_conll_data(
            docs=docs,
            docs_entities=docs_entities,
            max_len=int(config['BERT']['Max Sequence Length']),
            progress=True
        )

    del input_data_generator
    return input_vectors


read_input = config.getboolean('INPUT VECTORS', 'Read Input Vectors From Dir')
input_vectors = None
if not read_input:
    input_vectors = input_data_generation()


# -- Read or generate BERT input vectors, and other info ----------------------


@timer
def dataset_generation():
    read_input = config.getboolean('INPUT VECTORS', 'Read Input Vectors From Dir')
    use_balanced_dataset = config.getboolean('INPUT VECTORS', 'Use Balanced Dataset')

    # Recommended CoNLL split, reverse engineered to ratios.
    split_ratios = [0.6799, 0.1557, 0.1644]
    use_default_split = config.getboolean('TRAINING', 'Use Default Split')
    if not use_default_split:
        split_ratios = [float(config['TRAINING']['Training Set Size']),
                        float(config['TRAINING']['Validation Set Size']),
                        float(config['TRAINING']['Test Set Size'])]

    dataset_generator = DatasetGenerator()
    # If input vectors were generated in previous step,
    #  and no instructions to read from file
    if not read_input and input_vectors:
        dataset_generator = DatasetGenerator(*input_vectors)

    # Use balanced dataset
    if use_balanced_dataset:
        balanced_dataset_dir = config['INPUT VECTORS']['Balanced Dataset Dir']
        # Reading previously generated dataset from file
        if read_input and balanced_dataset_dir:
            print("Reading balanced dataset from files ...")
            dataset_generator.read_balanced_dataset(balanced_dataset_dir)
        # Generate balanced dataset
        else:
            n_neg_samples = config['INPUT VECTORS']['N Negative Samples']
            n_neg_samples = int(n_neg_samples) if n_neg_samples else 1

            print(f"Generating balanced dataset with ratio 1:{n_neg_samples} ...")
            dataset_generator.get_balanced_dataset(docs_entities, n_neg_samples)
            if balanced_dataset_dir:
                print("Writing balanced dataset to files ...")
                dataset_generator.write_balanced_dataset_to_files(balanced_dataset_dir)
        print("Splitting dataset ...")
        dataset_generator.get_split_dataset(split_ratios, dataset='balanced')

        dataset_to_doc = dataset_generator.balanced_dataset_to_doc
        dataset_to_mention = dataset_generator.balanced_dataset_to_entity
        dataset_to_candidate = dataset_generator.balanced_dataset_to_candidate
    # Use full dataset
    else:
        vec_dir = config['INPUT VECTORS']['Input Vectors Dir']
        # Reading previously generated dataset from file
        if read_input and vec_dir:
            print("Reading vectors ...")
            dataset_generator.read_from_directory(vec_dir)
        print("Splitting dataset ...")
        dataset_generator.get_split_dataset(split_ratios, dataset='full', docs_entities=docs_entities)

        dataset_to_doc, dataset_to_mention, dataset_to_candidate = dataset_generator.get_dataset_to_x(docs_entities)

    print("Getting DataLoaders ...")
    train_loader, val_loader, test_loader = \
        dataset_generator.get_data_loaders(batch_size=int(config['TRAINING']['Batch Size']))

    return train_loader, val_loader, test_loader, dataset_to_doc, dataset_to_mention, dataset_to_candidate


train_loader, val_loader, test_loader, dataset_to_doc, dataset_to_mention, dataset_to_candidate = dataset_generation()


# -- Generate BERT model ------------------------------------------------------


@timer
def model_generation():
    model_dir = config['BERT']['Bert Model Dir']
    if model_dir:
        model = load_bert_from_file(model_dir)
    else:
        model_path = config['BERT']['Model ID']
        model = BertBinaryClassification.from_pretrained(model_path, use_cls=False)

    freeze_n_transformers = config['TRAINING']['Freeze N Transformers']
    freeze_n_transformers = int(freeze_n_transformers) if freeze_n_transformers else 12
    model.freeze_n_transformers(freeze_n_transformers)

    return model

model = model_generation()


# -- Train and test -----------------------------------------------------------


@timer
def training():
    epochs = int(config['TRAINING']['Epochs'])
    save_dir = config['BERT']['Save Model Dir']

    train_update_freq = int(config['VERBOSITY']['Training Update Frequency'])
    validation_update_freq = int(config['VERBOSITY']['Validation Update Frequency'])
    test_update_freq = int(config['VERBOSITY']['Test Update Frequency'])

    train = True if epochs > 0 else False
    try:
        read_result_and_evaluate()
    except FileNotFoundError as e:
        print(e)
        print("Continuing to train and test procedure.")
        train = True

    if train:
        handler = ModelTrainer(model, train_loader, val_loader, test_loader, epochs)
        training_stats = handler.train(train_update_freq, validation_update_freq)
        save_bert_to_file(model, save_dir)
        plot_training_stats(training_stats, save_dir)
        handler.test(dataset_to_doc, dataset_to_mention, test_update_freq, dataset_to_candidate)


training()
