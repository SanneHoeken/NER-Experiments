from utils import load_json
import pandas as pd
import sys

def engineer_features(inputfile):
    """
    This function takes an inputfile that contains at least a column with tokens with header name 'token',
    and a column name with ner labels with header name 'ner', and enriches this file with several features for each token. 

    :param inputfile: path to inputfile
    :type inputfile: string
    
    """
    features = []
    data = pd.read_csv(inputfile, sep='\t', keep_default_na=False)
    for i in range(len(data)):
        feature_dict = dict()

        # Get features from current token
        token = data.loc[i, 'token']
        feature_dict['token'] = token
        feature_dict['pos'] = data.loc[i, 'pos']
        feature_dict['chunk'] = data.loc[i, 'chunk']      
        feature_dict['is_title'] = token.istitle()
        feature_dict['is_upper'] = token.isupper()  
        feature_dict['is_lower'] = token.islower() 

        # Get features from previous token
        previous_token = data.loc[i-1, 'token'] if i > 0 else ''
        feature_dict['previous_token'] = previous_token
        feature_dict['previous_pos'] = data.loc[i-1, 'pos'] if i > 0 else ''
        feature_dict['previous_chunk'] = data.loc[i-1, 'chunk'] if i > 0 else ''
        feature_dict['previous_is_title'] = previous_token.istitle()
        feature_dict['previous_is_upper'] = previous_token.isupper()  
        feature_dict['previous_is_lower'] = previous_token.islower()

        # Get features from next token
        next_token = data.loc[i+1, 'token'] if i < (len(data) - 1) else ''
        feature_dict['next_token'] = next_token
        feature_dict['next_pos'] = data.loc[i+1, 'pos'] if i < (len(data) - 1) else ''
        feature_dict['next_chunk'] = data.loc[i+1, 'chunk'] if i < (len(data) - 1) else ''
        feature_dict['next_is_title'] = next_token.istitle()
        feature_dict['next_is_upper'] = next_token.isupper()  
        feature_dict['next_is_lower'] = next_token.islower()

        features.append(feature_dict)
    
    # Append ner labels to dataframe and write to file
    features_df = pd.DataFrame(features)
    data_features = pd.concat([features_df, data['ner']], axis=1)
    data_features.to_csv(inputfile, sep='\t')


def main():
    
    data_infopath = sys.argv[1]
    data_info = load_json(data_infopath)
    gold_file = data_info['gold']['file']
    training_file = data_info['training']['file']
    engineer_features(gold_file)
    engineer_features(training_file)


if __name__ == "__main__":
    main()