from scipy.io.arff import loadarff
import pandas as pd
from config_loader import load
import argparse
from MyPreprocessing import MyPreprocessing
from MyCN2 import MyCN2
import sys
from time import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

##

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
    path = 'datasets/' + dataset + '.arff'
    try:
        data, meta = loadarff(path)
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

    ## Preprocessing
    preprocess = MyPreprocessing(bins_no=bins_no)
    preprocess.fit(data)
    df = preprocess.new_df
    labels = preprocess.labels_

    x_train, x_test, y_train, y_test = train_test_split(
        df , labels, train_size=train_percentage, random_state=42, stratify=labels)
    print(x_test.head())
    print(y_test.head())
    cn2 = MyCN2(beam_width=beam_width,
                min_significance=min_significance,
                negate=negate,
                disjunctive=disjunctive)
    start = time()
    cn2.fit(x_train)
    print('duration', time()-start)

    pred = cn2.predict(x_test, y_test)
    print('Precision, Recall, F-Score:')
    print(precision_recall_fscore_support(y_test.values, pred.values))
    print('Accuracy')
    print(accuracy_score(y_test.values, pred.values))
