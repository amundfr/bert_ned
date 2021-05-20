from configparser import ConfigParser
from src.conll_candidates_generator import ConllCandidatesGenerator
from src.input_data_generator import InputDataGenerator
from src.dataset_generator import DatasetGenerator

if __name__ == '__main__':
    print("\nStarting input data generation."
          "\nThis script relies on paths in 'config.ini'")

    print("\n 1. Getting 'config.ini'")
    config = ConfigParser()
    config.read('config.ini')

    print("\n 2. Getting Candidate Generator")
    candidate_generator = ConllCandidatesGenerator(
            spacy_nlp_vocab_dir=config['DATA']['Spacy Vocab Dir'],
            spacy_kb_file=config['DATA']['Spacy KB']
        )

    candidate_info_path = config['DATA']['Candidate Info']
    
    print(f"\n 3.a Trying to read candidate info file at '{candidate_info_path}'")
    docs_entities = []
    try:
        docs_entities = candidate_generator.read_entities_info(config['DATA']['Candidate Info'])
    except FileNotFoundError:
        print(" Could not find candidate info.")
        conll_file = config['DATA']['Conll Annotated']
        print(f"\n 3.b Generating candidates for annotated CoNLL file at '{conll_file}'")
        docs_entities = candidate_generator.get_docs_entities(conll_file)
        print(f"\n 3.c Writing candidate info to file '{candidate_info_path}'")
        candidate_generator.write_entities_info(candidate_info_path)

    docs = candidate_generator.get_docs(config['DATA']['Conll Annotated'])

    print("\n 4. Getting Input Data Generator")
    input_data_generator = InputDataGenerator(
            wikipedia_abstracts_file=config['DATA']['Wikipedia Abstracts'],
            tokenizer_pretrained_id=config['BERT']['Model ID']
        )

    print("\n 5. Generating input vectors")
    input_vectors = input_data_generator.generate_for_conll_data(
            docs=docs,
            docs_entities=docs_entities,
            max_len=int(config['BERT']['Max Sequence Length']),
            progress=True
        )

    vector_dir = config['INPUT VECTORS']['Input Vectors Dir']
    print(f"\n 6. Writing vectors to directory {vector_dir}")

    dataset_generator = DatasetGenerator(*input_vectors)
    dataset_generator.write_to_files(vector_dir)
