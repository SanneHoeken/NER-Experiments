from utils import load_json, dump_json, append_column_and_write_file
from classifier import TextClassifier
import sys
import os


def train_test_system(features_names_list, data_info_file, system, output_filename):
    """
    This function trains a classifier based on given system and data information,
    tests this classifier system and writes the predictions to an outputfile.

    :param features_names_list: list of indications of all feature columns that should be used
    :param data_info_file: path to file containing info about all necessary data
    :param system: name of the ML algorithm that is passed to the classifier 
    :param output_filename: path to conll outputfile
    :type feature_names_list: list
    :type data_info_file: string
    :type system: string
    :type output_filename: string
    """
    data = load_json(data_info_file)
    
    # Train model
    inputfile = data['training']['file']
    annotation_column = data['training']['annotation_column']
    model = TextClassifier(system)
    model.train(inputfile, features_names_list, annotation_column)

    # Classify
    gold_file = data['gold']['file']
    predictions = model.predict(gold_file)

    # Write output
    append_column_and_write_file(output_filename, gold_file, predictions, 'predictions')

    # Update data info
    name = os.path.basename(output_filename[:-6]) 
    data[name] = {'annotation_column': 'predictions', 'file': output_filename}
    dump_json(data_info_file, data)


def main():

    args = sys.argv
    system = args[2]
    output_filename = args[3]
    data_info_file = args[1]

    features_names_list = ['token', 'pos', 'chunk', 'is_title', 'is_upper', 'is_lower', \
        'previous_token', 'previous_pos', 'previous_chunk', 'previous_is_title',\
        'previous_is_upper', 'previous_is_lower', 'next_token', 'next_pos', \
        'next_chunk', 'next_is_title', 'next_is_upper', 'next_is_lower']

    train_test_system(features_names_list, data_info_file, system, output_filename)


if __name__ == "__main__":
    
    main()