import csv, json
import pandas as pd


def load_json(json_file):
    """
    Opens json-file and returns it as dictionary

    :param json_file: path to json-file
    :type json-file: string
    
    :returns: the content of the json file as dictionary
    """
    with open(json_file) as infile:
        content = json.load(infile)
    
    return content


def dump_json(json_file, dictionary):
    """
    Opens json-file and writes dictionary

    :param json-file: path to json-file
    :param dictionary: dictionary that should be dumped
    :type json-file: string
    :type dictionary: dict
    """
    with open(json_file, "w") as outfile:
        json.dump(dictionary, outfile)


def extract_column(inputfile, column, delimiter='\t'):
    '''
    This function extracts the values in a column from a file
    
    :param inputfile: the path to the file
    :param column: header name of column
    :param delimiter: optional parameter to overwrite the default delimiter (tab)
    :type inputfile: string
    :type column: string
    :type delimiter: string

    :returns: the column values as a list
    '''
    conll_input = pd.read_csv(inputfile, sep='\t')
    values = conll_input[column].tolist()
    return values


def append_column_and_write_file(outputfile, inputfile, column, column_name):
    """
    This function takes a tab seperated file writes to outputfile with an added column

    :param outputfile: path to outputfile
    :param inputfile: path to inputfile
    :param column: a list, array or Series dat should be added
    :param column_name: indication of the column
    :type outputfile: string
    :type inputfile: string
    :type column: list, array or Series
    :type column_name: string
    """
    input_df = pd.read_csv(inputfile, sep='\t')
    input_df[column_name] = column
    input_df.to_csv(outputfile, sep='\t')
    