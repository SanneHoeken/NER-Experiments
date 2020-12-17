from utils import extract_column
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from sklearn_crfsuite import CRF
import pandas as pd
import numpy as np
import gensim.downloader as api

class TextClassifier():
    """
    Class that can be trained to different classifier systems based on different features
    """

    def __init__(self, modelname):

        self.modelname = modelname
        self.model = None
        self.vectorizer = None
        self.features_names_list = []

        
    def train(self, inputfile, features_names_list, annotation_column):
        """
        This function fits a classification model as specified on training data

        :param inputfile: path to inputfile containing the training data
        :param features_names_list: list of indications of all feature columns that should be used
        :param annotation_column: indication of column with annotations
        :type inputfile: string
        :type features_names_list: list
        :type annotation_column: string
        """

        # initialize the right model
        if self.modelname == 'logreg':
            self.model = LogisticRegression()

        elif self.modelname == 'naivebayes':
            self.model = BernoulliNB()

        elif self.modelname == 'svm':
            self.model = LinearSVC()

        elif self.modelname == 'crf':
            self.model = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)

        # store features_names_list as class attribute
        self.features_names_list = features_names_list

        # get training features and labels
        train_features = self.get_features(inputfile)
        train_targets = self.get_labels(inputfile, annotation_column)

        # fit the model
        self.model.fit(train_features, train_targets)


    def predict(self, inputfile):
        """
        This function classifies the observations of the given inputfile

        :param inputfile: path to inputfile
        :type inputfile: string
        
        :returns: list of predictions
        """
        # get features and predict the labels
        features = self.get_features(inputfile)
        predictions = self.model.predict(features)

        # join the predictions for every sentence if the model is CRF
        if self.modelname == 'crf':
            joint_preds = [pred for sent in predictions for pred in sent]
            return joint_preds

        return predictions

    
    def get_features(self, inputfile):
        """
        This function extracts all features transformed to vector from the inputfile according to the model and feature settings

        :param inputfile: path to inputfile
        :type inputfile: string
        
        :returns: vectors that represent the features
        """
        # extract and return features if the model is a CRF
        if self.modelname == 'crf':
            return self.extract_crf_features(inputfile)

        # extract all features as one-hot representations except for embeddings (if specified)
        sparse_features = None
        sparse_features_list = [feature for feature in self.features_names_list if feature != 'embedding']
        if sparse_features_list:
            sparse_features = self.extract_features(inputfile, sparse_features_list)

        # extract word embedding representations if specified in features list
        if 'embedding' in self.features_names_list:
            embeddings = self.extract_embeddings_as_features(inputfile)
            
            # combine embedding with sparse features if both should be represented, and return them
            if sparse_features != None:
                combined_features = self.combine_sparse_and_dense_features(embeddings, sparse_features) 
                return combined_features        
            else:
                return embeddings
    
        return sparse_features


    def get_labels(self, inputfile, annotation_column):
        """
        This function extracts all the labels from the inputfile in the form required by the model

        :param inputfile: path to inputfile
        :param annotation_column: indication of column with annotations
        :type inputfile: string
        :type annotation_column: string
        
        :returns: labels in required form
        """

        # extract the labels in the form required for CRF and return
        if self.modelname == 'crf':
            return self.extract_crf_labels(inputfile, annotation_column)

        # extract and return the labels in the form required for all other models
        return extract_column(inputfile, annotation_column)

        
    def extract_features(self, inputfile, features):
        """
        This function extracts the specified features and transforms them to vectors

        :param inputfile: path to tab seperated inputfile with header
        :param features: list of feature names that should be extracted
        :type inputfile: string
        :type features: list
        
        :returns: feature vectors
        """
        # reads the inputfile 
        data = pd.read_csv(inputfile, sep='\t', keep_default_na=False)
        # converts the specified feature columns to a list of dictionaries
        features = data[features].to_dict('records')
        
        # transforms the feature dictionaries to vectors, and stores the fitted vectorizer
        if not self.vectorizer:
            self.vectorizer = DictVectorizer()
            self.vectorizer.fit(features)

        features = self.vectorizer.transform(features)
        
        return features


    def extract_embeddings_as_features(self, inputfile, embedding_model='word2vec-google-news-300'):
        '''
        Function that extracts features and gold labels using word embeddings
        
        :param inputfile: path to inputfile
        :param token_column: header name of column with tokens
        :param embedding_model: name of a pretrained word embedding model
        :type inputfile: string
        :type token_column: string
        :type embedding_model: string
        
        :return features: list of vector representation of tokens
        '''
        ### This code was partially inspired by code included in the HLT course, obtained from https://github.com/cltl/ma-hlt-labs/, accessed in May 2020.
        features = []
        embedding_model = api.load(embedding_model)
        tokens = extract_column(inputfile, 'token')
        
        for token in tokens:
            if token in embedding_model:
                vector = embedding_model[token]
            else:
                vector = [0]*300
            features.append(vector)

        return features


    def combine_sparse_and_dense_features(self, dense_vectors, sparse_features):
        '''
        Function that takes sparse and dense feature representations and appends their vector representation
        
        :param dense_vectors: list of dense vector representations
        :param sparse_features: list of sparse vector representations
        :type dense_vector: list of arrays
        :type sparse_features: list of lists
        
        :returns: list of arrays in which sparse and dense vectors are concatenated
        '''
        ### SOURCE: code included in the HLT course, obtained from https://github.com/cltl/ma-hlt-labs/.
        
        combined_vectors = []
        sparse_vectors = np.array(sparse_features.toarray())
        
        for index, vector in enumerate(sparse_vectors):
            combined_vector = np.concatenate((vector, dense_vectors[index]))
            combined_vectors.append(combined_vector)
        
        return combined_vectors


    def extract_crf_features(self, inputfile):
        """
        This function extracts the features for every token grouped for every sentence

        :param inputfile: path to inputfile
        :type inputfile: string
        
        :returns: a list with for every sentence a list of feature dictionaries
        """
        ### based on: https://github.com/TeamHG-Memex/sklearn-crfsuite/blob/master/docs/CoNLL2002.ipynb

        # reads the inputfile 
        data = pd.read_csv(inputfile, sep='\t', keep_default_na=False)
        # converts the specified feature columns to a list of dictionaries
        features = data[self.features_names_list].to_dict('records')

        # appends for every sentence ('.' as sentence boundary) the feature dictionaries to a list 
        # appends that list if complete to a list for all sentences
        sentences = []
        sentence = []

        for token_dict in features:
            sentence.append(token_dict)
            if token_dict['token'] == '.':
                sentences.append(sentence)
                sentence = []
        sentences.append(sentence)
        
        return sentences


    def extract_crf_labels(self, inputfile, annotation_column):
        """
        This function extracts the label for every token grouped for every sentence

        :param inputfile: path to inputfile
        :type inputfile: string
        
        :returns: a list with for every sentence a list of labels
        """
        ### based on: https://github.com/TeamHG-Memex/sklearn-crfsuite/blob/master/docs/CoNLL2002.ipynb

        # reads the inputfile 
        data = pd.read_csv(inputfile, sep='\t', keep_default_na=False)
        # converts the specified feature columns to a list of dictionaries
        data_dicts = data.to_dict('records')

        # appends for every sentence ('.' as sentence boundary) the labels to a list 
        # appends that list if complete to a list for all sentences
        sentences_labels = []
        sentence_labels = []

        for token_dict in data_dicts:
            sentence_labels.append(token_dict[annotation_column])
            if token_dict['token'] == '.':
                sentences_labels.append(sentence_labels)
                sentence_labels = []
        sentences_labels.append(sentence_labels)

        return sentences_labels


