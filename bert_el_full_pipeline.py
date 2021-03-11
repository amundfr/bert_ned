from configparser import ConfigParser
import time

from src.conll_candidates_generator import ConllCandidatesGenerator
from src.input_data_generator import InputDataGenerator
from src.dataset_generator import DatasetGenerator
from src.bert_model import load_bert_from_file, save_bert_to_file, get_class_weights_tensor
from src.trainer import ModelTrainer, plot_training_stats

import torch
import numpy as np
import random

# Set the seed value everywhere to make this reproducible
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


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

# --------------------------------------------------


@timer
def candidate_generation(config: ConfigParser):
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


docs_entities, docs = candidate_generation(config)

# --------------------------------------------------


@timer
def input_data_generation(config: ConfigParser):
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


# input_vectors = input_data_generation(config)

# --------------------------------------------------

dataset_generator = DatasetGenerator()

@timer
def dataset_generation(config: ConfigParser):
    vec_dir = config['INPUT VECTORS']['Input Vectors Dir']
    split_ratios = [float(config['TRAINING']['Training Set Size']),
                    float(config['TRAINING']['Validation Set Size']),
                    float(config['TRAINING']['Test Set Size'])]
    balanced_dataset_dir = config['INPUT VECTORS']['Balanced Dataset Dir']
    print("Reading vectors ...")
    dataset_generator.read_from_directory(vec_dir)
    print("Making balanced dataset ...")
    dataset_generator.get_balanced_dataset(docs_entities)
    print("Writing balanced dataset to files ...")
    dataset_generator.write_balanced_dataset_to_files(balanced_dataset_dir)
    print("Reading balanced dataset from files ...")
    dataset_generator.read_balanced_dataset(balanced_dataset_dir)
    print("Splitting dataset ...")
    dataset_generator.get_split_dataset(split_ratios, dataset='balanced')
    print("Getting DataLoaders ...")
    return dataset_generator.get_data_loaders(batch_size=int(config['TRAINING']['Batch Size']))


train_loader, val_loader, test_loader = dataset_generation(config)
neg, pos = dataset_generator.get_dataset_balance_info()
# These two are needed to take accuracy over entities, rather than over candidates
dataset_to_doc = dataset_generator.balanced_dataset_to_doc
dataset_to_entity = dataset_generator.balanced_dataset_to_entity
dataset_generator = None

# --------------------------------------------------

@timer
def model_generation(config: ConfigParser):
    model_dir = config['BERT']['Bert Model Dir']
    model = load_bert_from_file(model_dir)
    # TODO: test use of class weights on accuracy performance
    model.set_class_weights(get_class_weights_tensor(1, 1))
    model.freeze_n_transformers(8)
    return model

model = model_generation(config)

# --------------------------------------------------

@timer
def training(config: ConfigParser):
    epochs = int(config['TRAINING']['Epochs'])
    save_dir = config['BERT']['Save Model Dir']
    train_update_freq = int(config['VERBOSITY']['Training Update Frequency'])
    validation_update_freq = int(config['VERBOSITY']['Validation Update Frequency'])
    test_update_freq = int(config['VERBOSITY']['Test Update Frequency'])
    handler = ModelTrainer(model, train_loader, val_loader, test_loader, epochs)
    training_stats = handler.train(train_update_freq, validation_update_freq)
    handler.test(dataset_to_doc, dataset_to_entity, test_update_freq)
    plot_training_stats(training_stats, save_dir)
    save_bert_to_file(model, save_dir)


training(config)
