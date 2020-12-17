# NER-Experiment

This project is part of the master course Machine Learning for NLP, which is part of the Humanities Research master at VU University Amsterdam.
October 2020.

### Project

In this project I experiment with different Named Entity Recognition systems.

## Getting started

### Requirements

This codebase is written entirely in Python 3.7. requirements.txt contains all necessary packages to run the code successfully. These are easy to install via pip using the following instruction:

```
pip install -r requirements.txt
```

Or via conda:

```
conda install --file requirements.txt
```

### Structure

The following list describes the folders in the project:

- **/code**: contains all code of this project
- **/settings**: contains json-files with info for running code

### Get data and adapt data info file to your own settings

This system experiment setup is made to be trained on data taken from the Reuters Corpus and to be tested on the data provided trough the CoNLL shared task of 2003. In addition, in this setup it is possible to compare output of the systems with output on the same gold data from Stanford CoreNLP and Spacy. In the **/settings** folder a json-file named **data_info.json** should be used to provide the information about the location and format of these data. As can be seen, there is a default given. 

#### Default

You can use the default content of the json file if you create a **/data** directory in the same directory in which the **/code** and **/settings** directories are located, in which the following files are stored:
- gold_stripped.conll
- reuters-train-tab-stripped.en
- spacy_out_matched_tokens.conll
- stanford_out_matched_tokens.conll

#### Adjusted 

If your enviroment does not match the default settings, adjust the file to your data settings. The 'gold' and 'training' elements must at least be present and each element must have the attributes 'file' (path to the data file), 'header' (list of names describing the columns, this header will be inserted at the preprocessing step), 'annotation_column' (index of the column containing the labels) and 'extension' (file-extension).

## Using the main programs

First of all, change the working directory of the execution environment to the **/code** directory.

1. **Preprocessing data.**

  This program can be run by calling:

  ```
  python preprocessing.py [path to data info] [path to file with conversions]
  ```

  The path to data info should point to the json-file that provides the information of the data, as explained above. Probably: '../settings/data_info.json'

  The path to file with conversions should point to a json-file that specifies all the label conversions that should be made for each data-file. See **/settings/conversions.json** for an example (which I highly recommend to use).
  
  After the execution of the program, preprocessed versions of the datafiles will be stored in the same folder as where they were found and an updated version of the data information will be stored in a json-file in **/settings** as **data_info-preprocessed.json**.

2. **Feature engineering.**

  This program can be run by calling:

  ```
  python feature_engineering.py [path to preprocessed data info]
  ```

  The path to preprocessed data info should point to the json-file that provides for every preprocessed datafile the information as described above, probably: '../settings/data_info-preprocessed.json'.

  After the execution of the program, every token in the gold and training file will be enriched with a selection of features. 

3. **Training and testing of a NER system.**

  This program can be run by calling:

  ```
  python system_experiment.py [path to preprocessed data info] [name of machine learning algorithm] [path to conll outputfile]
  ```

  The path to preprocessed data info should point to the json-file that provides for every preprocessed datafile the information as described above, probably: '../settings/data_info-preprocessed.json'.
  
  The name of the machine learning algorithm can be one of the following: 'logreg', 'naivebayes', 'svm' or 'crf'. 
  
  The path to the conll outputfile should point to an existing directory and the name of the file should end with the '.conll' extension.
  
  After the execution of the program, the output of the trained and tested NER system will be stored in the specified outputfile. In addition, the provided json-file with data information is updated with the information of the generated output file.

4. **Evaluating the systems.**

  This program can be run by calling:

  ```
  python evaluation.py [path to preprocessed data info]
  ```

  The path to preprocessed data info should point to a json-file that provides for every preprocessed datafile the information as described above, probably: '../settings/data_info.json-preprocessed'.
  
  The program evaluates all the systems, of which the outputfile is given in the json-file with data information, and prints the results.

**Example**

If your environment meets the default settings, an experiment with an SVM classifier would be possible with running the following command lines one after the other:

```
python preprocessing.py '../settings/data_info.json' '../settings/conversions.json'
```

```
python feature_engineering.py '../settings/data_info-preprocessed.json'
```

```
python system_experiment.py '../settings/data_info-preprocessed.json' 'svm' '../data/svm_out.conll'
```

```
python evaluation.py '../settings/data_info-preprocessed.json'
```

## Author
- Sanne Hoeken (student number: 2710599)