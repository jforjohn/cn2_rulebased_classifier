from scipy.io.arff import loadarff
import pandas as pd
from config_loader import load
import argparse
from MyPreprocessing import MyPreprocessing
from MyCN2 import MyCN2
import sys
from time import time
from os import path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

##
def rules2file(file_path, rules):
    with open(file_path, 'w') as fd:
        if cn2.disjunctive:
            liaison = 'or'
        else:
            liaison = 'and'

        fd.write('if ')
        # first rule
        for tup in rules.loc[0,'rule'][:-1]:
            att, val, negate = tup
            if negate:
                fd.write(f'{att} != {val} {liaison}')
            else:
                fd.write(f'{att} == {val} {liaison}')
        att, val, negate = rules.loc[0,'rule'][-1]
        if negate:
            fd.write(f'{att} != {val} ')
        else:
            fd.write(f'{att} == {val} ')
        pred = rules.loc[0, 'prediction']
        fd.write(f'then\n  {pred}\n')

        # all rules following except the last one which is the defalut one
        for ind, row in rules.iloc[1:-1].iterrows():
            rule = row.loc['rule']
            fd.write('elif ')
            for tup in rule[:-1]:
                att, val, negate = tup
                if negate:
                    fd.write(f'{att} != {val} {liaison} ')
                else:
                    fd.write(f'{att} == {val} {liaison} ')
            att, val, negate = rules.loc[ind,'rule'][-1]
            if negate:
                fd.write(f'{att} != {val} ')
            else:
                fd.write(f'{att} == {val} ')
            pred = row.loc['prediction']
            fd.write(f'then {pred}\n')

        # default rule
        fd.write('else ')
        #default = rules.loc[rules.shape[0]-1,'rule'][0]
        #fd.write(f'{default} then ')
        pred = rules.loc[rules.shape[0]-1, 'prediction']
        fd.write(f'{pred}\n')

##
if __name__ == '__main__':
    # Loads config
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", default="cn2.cfg",
        help="specify the location of the clustering config file"
    )
    args, _ = parser.parse_known_args()

    config_file = args.config
    config = load(config_file)

    ##
    dataset = config.get('cn2', 'dataset')
    data_path = 'datasets/' + dataset + '.arff'
    try:
        data, meta = loadarff(data_path)
    except FileNotFoundError:
        print("Dataset '%s' cannot be found in the path %s" %(dataset, path))
        sys.exit(1)
    
    try:
        bins_no = config.getint('cn2', 'bins_no')
    except ValueError:
        print('bins_no should be integer, default: 5')
        bins_no = 5
    
    try:
        beam_width = config.getint('cn2', 'beam_width')
    except ValueError:
        print('beam_width should be integer, default: 3')
        beam_width = 3

    try:
        min_significance = config.getfloat('cn2', 'min_significance')
    except ValueError:
        print('min_significance should be float, default: 0.5')
        min_significance = 0.5

    try:
        negate = config.getboolean('cn2', 'negate')
    except ValueError:
        print('negate should be boolean or yes/no, default: no')
        negate = False

    try:
        disjunctive = config.getboolean('cn2', 'disjunctive')
    except ValueError:
        print('disjunctive should be boolean or yes/no, default: no')
        disjunctive = False
    
    try:
        train_percentage = config.getfloat('cn2', 'train_percentage')
    except ValueError:
        print('train_percentage should be float, default: 0.7')
        train_percentage = 0.7

    try:
        output_dir = config.get('cn2', 'output_dir')
    except ValueError:
        print('output_dir should be float, default: outputs')
        train_percentage = 'outputs'

    print("###")
    print(dataset)
    print("###")
    
    ## Preprocessing
    preprocess = MyPreprocessing(bins_no=bins_no)
    preprocess.fit(data)
    df = preprocess.new_df
    labels = preprocess.labels_

    try:
        x_train, x_test, y_train, y_test = train_test_split(
            df , labels, train_size=train_percentage, random_state=42, stratify=labels.values)
    except ValueError:
        # for the case of the least populated class in y to have only 1 member
        x_train, x_test, y_train, y_test = train_test_split(
            df , labels, train_size=train_percentage, random_state=42)

    cn2 = MyCN2(beam_width=beam_width,
                min_significance=min_significance,
                negate=negate,
                disjunctive=disjunctive)

    print('Train model')
    start = time()
    cn2.fit(df)
    print('No of selectors:', len(cn2.selectors))
    print('Train duration', time()-start)
    print()

    print('Test of the whole train model')
    print()
    start = time()
    pred = cn2.predict(df, labels)
    print('Test duration', time()-start)

    print('Precision, Recall, F-Score:')
    print(precision_recall_fscore_support(labels.values, pred.values))
    print()
    print('Unique labels:')
    print(labels.loc[:, 'Class'].unique())
    print()
    print('Precision, Recall, F-Score per label:')
    print(precision_recall_fscore_support(labels.values, pred.values, labels=labels.loc[:, 'Class'].unique()))
    print()
    print('Accuracy')
    print(accuracy_score(labels.values, pred.values))
    print()

    rules = cn2.df_rules.loc[:, ['rule','prediction']]
    file_path = path.join(output_dir, f'{dataset}-signficance-{str(cn2.min_significance)}-k-{str(cn2.beam_width)}-model.txt')
    rules2file(file_path, rules)
    print()

    print('------------------------------------')
    print()
    print('Train set')
    print()
    cn2 = MyCN2(beam_width=beam_width,
                min_significance=min_significance,
                negate=negate,
                disjunctive=disjunctive)
    start = time()
    cn2.fit(x_train)
    print('Train duration', time()-start)

    print()
    print('Test accuracy of the train set')

    rules = cn2.df_rules.loc[:, ['rule','prediction']]
    file_path = path.join(output_dir, f'{dataset}-significance-{str(cn2.min_significance)}-k-{str(cn2.beam_width)}-trainset-model.txt')
    rules2file(file_path, rules)

    start = time()
    pred = cn2.predict(x_train, y_train)
    print('Test duration', time()-start)
    print()
    
    print('Precision, Recall, F-Score:')
    print(precision_recall_fscore_support(y_train.values, pred.values))
    print()
    print('Unique labels:')
    print(y_train.loc[:, 'Class'].unique())
    print()
    print('Precision, Recall, F-Score per label:')
    print(precision_recall_fscore_support(y_train.values, pred.values, labels=y_train.loc[:, 'Class'].unique()))
    print()
    print('Accuracy')
    print(accuracy_score(y_train.values, pred.values))
    print()

    print('TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT')
    print()
    print('Test set')
    print()
    start = time()
    pred = cn2.predict(x_test, y_test)
    print('Test duration', time()-start)
    print()
    print('Precision, Recall, F-Score:')
    print(precision_recall_fscore_support(y_test.values, pred.values))
    print()
    print('Unique labels:')
    print(y_test.loc[:, 'Class'].unique())
    print()
    print('Precision, Recall, F-Score per label:')
    print(precision_recall_fscore_support(y_test.values, pred.values, labels=y_test.loc[:, 'Class'].unique()))
    print()
    print('Accuracy')
    print(accuracy_score(y_test.values, pred.values))
