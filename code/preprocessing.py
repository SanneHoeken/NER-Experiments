from utils import load_json, dump_json
import collections, csv, sys

def alligned(conllfile1, conllfile2):
    '''
    Check whether the tokens of two conll files are aligned
    
    :param conll1: path to the first conll file
    :param conll2: path to the second conll file
    
    :returns boolean indicating whether tokens match or not
    '''
    conll1 = read_in_datafile(conllfile1)
    conll2 = read_in_datafile(conllfile2)

    for row1, row2 in zip(conll1, conll2):
        if row1[0] != row2[0]:
            print(conllfile1, conllfile2, 'do not align')
            return False
    
    return True


def read_in_datafile(data_file, delimiter='\t'):
    '''
    Read in data file and return structured object
    :param data_file: path to data_file
    :param delimiter: specifies how columns are separated. Tabs are standard in conll
    
    :returns structured representation of information included in data file
    '''

    with open(data_file, 'r') as infile:
        file_as_csvreader = csv.reader(infile, delimiter=delimiter)
        file_object = [row for row in file_as_csvreader]

    return file_object


def create_converted_output(data_object, annotation_identifier, header, conversions, outputfilename):
    '''
    Check which annotations need to be converted for the output to match and convert them
    
    :param data_object: structured object with conll annotations
    :param annotation_identifier: indicator of how to find the annotations in the object (e.g. key of dictionary, index of list)
    :param conversions: dictionary with the conversions that apply
    :param outputfilename: path to preprocessed output file
    
    '''
    with open(outputfilename, 'w') as outputcsv:
        csvwriter = csv.writer(outputcsv, delimiter='\t')
        csvwriter.writerow(header)
        for row in data_object:
            annotation = row[annotation_identifier]
            if annotation in conversions:
                row[annotation_identifier] = conversions.get(annotation)
            csvwriter.writerow(row)


def preprocess_file(data_file, column_identifier, header, conversions, outputfilename):
    '''
    Guides the full process of preprocessing files and outputs the modified files.
    
    :param data_file: path to input file
    :param column_identifier: object providing the identifier for target column
    :param conversions: path to a file that defines conversions
    :param outputfilename: path to preprocessed output file
    '''
    data = read_in_datafile(data_file)
    conversions = load_json(conversions)
    create_converted_output(data, column_identifier, header, conversions, outputfilename)


def main():
    
    args = sys.argv

    # Load data info
    data = load_json(args[1])
    goldfile = data['gold']['file']

    # Iterate over data files
    for key, value in data.items():

        # Check allignment
        if key == "spacy" or key == "stanford":
            if not alligned(value['file'], goldfile):
                continue
        
        # Preprocess file
        outputfilename = value['file'].replace(value['extension'], '-preprocessed' + value['extension'])
        preprocess_file(value['file'], value['annotation_column'], value['header'], args[2], outputfilename)

        # Update data-information of preprocessed file
        value['file'] = outputfilename
        value['annotation_column'] = value['header'][value['annotation_column']]

    # Write updated data info to outfile
    dump_json(args[1].replace('.json', '-preprocessed.json'), data)


if __name__ == "__main__":
    main()

    