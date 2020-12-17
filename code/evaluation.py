import pandas as pd
from collections import defaultdict, Counter
from utils import extract_column, load_json
import sys

def get_counts(goldannotations, machineannotations):
    '''
    This function compares the gold annotations to machine output
    
    :param goldannotations: the gold annotations
    :param machineannotations: the output annotations of the system in question
    :type goldannotations: the type of the object created in extract_annotations
    :type machineannotations: the type of the object created in extract_annotations
    
    :returns: a countainer providing the counts for each predicted and gold class pair
    '''
    
    # SOURCE: https://stackoverflow.com/questions/49393683/how-to-count-items-in-a-nested-dictionary, last accessed 22.10.2020
    evaluation_counts = defaultdict(Counter)   
    for gold, pred in zip(goldannotations, machineannotations):
        evaluation_counts[gold][pred] += 1
        
    return evaluation_counts


def get_truefalse_posnegs(evaluation_counts):
    '''
    Calculates true and false positives and true and false negatives for each class and returns them in a dictionary
    
    :param evaluation_counts: a container from which you can obtain the true positives, false positives and false negatives for each class
    :type evaluation_counts: type of object returned by obtain_counts
    
    :returns true and false positives and true and flase negatives for each class in a container
    '''
    posnegs = dict()
    
    for label in evaluation_counts:
        gold_neg = [key for key in evaluation_counts if key != label]
        pred_neg = [key for key in evaluation_counts[label] if key != label]
        TP = evaluation_counts[label][label]
        FP = sum([evaluation_counts[x][label] for x in gold_neg])
        TN = sum([evaluation_counts[x][y] for x in gold_neg \
                  for y in [key for key in evaluation_counts[x] if key != label]])
        FN = sum([evaluation_counts[label][x] for x in pred_neg])
        posnegs[label] = {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}
    
    return posnegs


def calculate_metrics(evaluation_counts, goldannotations):
    '''
    Calculate precision recall and fscore for each class and return them in a dictionary
    
    :param evaluation_counts: a container from which you can obtain the true positives, false positives and false negatives for each class
    :param goldannotations: list of gold annotations
    :type evaluation_counts: type of object returned by obtain_counts
    :type goldannotations: list
    
    :returns the precision, recall and f-score of each class in a container
    '''
    metrics = dict()
    posnegs = get_truefalse_posnegs(evaluation_counts)
    
    # Calculate precision, recall and f-score for each class
    for label in posnegs:
        try:
            precision = posnegs[label]['TP']/(posnegs[label]['TP']+posnegs[label]['FP'])
        except ZeroDivisionError:
            precision = 0
        try:
            recall = posnegs[label]['TP']/(posnegs[label]['TP']+posnegs[label]['FN'])
        except ZeroDivisionError:
            recall = 0
        try:
            fscore = 2 * posnegs[label]['TP']/(2 * posnegs[label]['TP'] + posnegs[label]['FP'] + posnegs[label]['FN'])
        except ZeroDivisionError:
            fscore = 0
        metrics[label] = {'precision': precision, 'recall': recall, 'f-score': fscore}
    
    # Calculate micro average precision, recall and f-score for all classes
    totalTP = sum([posnegs[label]['TP'] for label in posnegs])
    totalFP = sum([posnegs[label]['FP'] for label in posnegs])
    totalFN = sum([posnegs[label]['FN'] for label in posnegs])
    micro_avg_precision = totalTP / (totalTP + totalFP)
    micro_avg_recall = totalTP / (totalTP + totalFN)
    micro_avg_fscore = 2 * totalTP / (2 * totalTP + totalFP + totalFN)
    metrics['micro_average'] = {'precision': micro_avg_precision, 'recall': micro_avg_recall, 'f-score': micro_avg_fscore}

    # Calculate macro average precision, recall and f-score for all classes
    macro_avg_precision = sum([metrics[label]['precision'] for label in posnegs])/len(posnegs)
    macro_avg_recall = sum([metrics[label]['recall'] for label in posnegs])/len(posnegs)
    macro_avg_fscore = sum([metrics[label]['f-score'] for label in posnegs])/len(posnegs)
    metrics['macro_average'] = {'precision': macro_avg_precision, 'recall': macro_avg_recall, 'f-score': macro_avg_fscore}

    # Calculate weighted average precision, recall and f-score for all classes
    labelcounts = pd.Series(goldannotations).value_counts()
    totalcount = pd.Series(goldannotations).count()
    weighted_avg_precision = sum([metrics[label]['precision'] * labelcounts[label] for label in posnegs])/totalcount
    weighted_avg_recall = sum([metrics[label]['recall'] * labelcounts[label] for label in posnegs])/totalcount
    weighted_avg_fscore = sum([metrics[label]['f-score'] * labelcounts[label] for label in posnegs])/totalcount
    metrics['weighted_average'] = {'precision': weighted_avg_precision, 'recall': weighted_avg_recall, 'f-score': weighted_avg_fscore}
    
    return metrics


def provide_confusion_matrix(evaluation_counts):
    '''
    Reads in the evaluation counts and provide a confusion matrix for each class
    
    :param evaluation_counts: a container from which you can obtain the true positives, false positives and false negatives for each class
    :type evaluation_counts: type of object returned by obtain_counts
    
    :prints out a confusion matrix
    '''
    
    confusion_matrix = pd.DataFrame.from_dict(evaluation_counts, orient='index')
    confusion_matrix = confusion_matrix.fillna(0)
    print(confusion_matrix)


def carry_out_evaluation(gold_annotations, system_annotations):
    '''
    Carries out the evaluation process (from input file to calculating relevant scores)
    
    :param gold_annotations: list of gold annotations
    :param systemfile: path to file with system output
    :param systemcolumn: indication of column with relevant information
    :param delimiter: specification of formatting of file (default delimiter set to '\t')
    
    returns evaluation information for this specific system
    '''
    evaluation_counts = get_counts(gold_annotations, system_annotations)
    provide_confusion_matrix(evaluation_counts)
    evaluation_outcome = calculate_metrics(evaluation_counts, gold_annotations)
    
    return evaluation_outcome


def provide_output_tables(evaluations):
    '''
    Create tables based on the evaluation of various systems
    
    :param evaluations: the outcome of evaluating one or more systems
    '''
    # SOURCE: https:stackoverflow.com/questions/13575090/construct-pandas-dataframe-from-items-in-nested-dictionary
    evaluations_pddf = pd.DataFrame.from_dict({(i,j): evaluations[i][j]
                                              for i in evaluations.keys()
                                              for j in evaluations[i].keys()},
                                             orient='index')
    print(evaluations_pddf)


def run_evaluations(data_infofile):
    '''
    Carry out standard evaluation for one or more system outputs
    
    :param goldfile: path to file with goldstandard
    :param data_infofile: filepath to json providing information about system and data to files
    :type goldfile: string
    :type data_infofile: string
    
    :returns the evaluations for all systems
    '''
    evaluations = {}
    data = load_json(data_infofile)
    gold_annotations = extract_column(data['gold']['file'], data['gold']['annotation_column'])
    for key, value in data.items():
        if key != 'training' and key != 'gold':
            system_annotations = extract_column(value['file'], value['annotation_column'])
            sys_evaluation = carry_out_evaluation(gold_annotations, system_annotations)
            evaluations[key] = sys_evaluation
    return evaluations


def main():

    args = sys.argv
    evaluations = run_evaluations(args[1])
    provide_output_tables(evaluations) 


if __name__ == "__main__":
    main()


