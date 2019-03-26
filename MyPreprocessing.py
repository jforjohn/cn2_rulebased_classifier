#from scipy.io.arff import loadarff
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

class MyPreprocessing(BaseEstimator, TransformerMixin):
    def __init__(self, bins_no=5):
        self.bins_no = bins_no
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)

    def fit(self, data):
        
        df = pd.DataFrame(data)
        df = df.replace(b'?', None)
        # rename class column to be consistent
        df.columns = df.columns[:-1].values.tolist() + ['Class']

        # get label
        labels = df.iloc[:, -1]
        if labels.dtype == np.object:
            self.labels_ = pd.DataFrame(labels.apply(
                lambda x: x.decode('utf-8')))
        else:
            self.labels_ = pd.DataFrame(labels.apply(str))
        # remove labels
        df = df.drop(df.columns[len(df.columns) - 1], axis=1)
        #nan_cols = df.loc[:, df.isna().any()].columns

        df_obj = df.select_dtypes(include='object')
        # handle byte numerical data
        for col in df_obj.columns:
            try:
                int_col = df_obj.loc[:,col].apply(np.float64)
                df_obj = df_obj.drop([col], axis=1)
                df.loc[:, col] = int_col
            except ValueError:
                df_obj.loc[:, col] = df_obj.loc[:,col].apply(lambda x: x.decode('utf-8'))
                df_obj.loc[:, col] = df_obj.loc[:,col].fillna(df_obj.loc[:,col].mode()[0])
            
        df_num = df.select_dtypes(exclude='object')
        df_num_process = pd.DataFrame()

        #df_num = df_num.replace(np.NaN, 0)
        for col in df_num.columns:
            df_num.loc[:, col] = df_num.loc[:,col].fillna(df_num.loc[:, col].mean())
            col_discrete = pd.DataFrame(
                pd.cut(df_num.loc[:, col],
                self.bins_no,
                labels=[f'{col}_{i+1}' for i in range(self.bins_no)], 
                duplicates='drop'), columns=[col])
            df_num_process = pd.concat([df_num_process, col_discrete], axis=1)

        self.new_df = pd.concat(
            [df_obj, df_num_process, self.labels_],
            axis=1, sort=False)

#
#print(df.select_dtypes(exclude='object'))
#print(df.select_dtypes(include='object'))
#plt.interactive(False)
#plt.show(block=True)


##
#print(agg_clustering(df_preprocess, 'Single', 3))
#agg = AgglomerativeClustering(n_clusters=2, linkage='complete')
#print(agg.fit_predict(df_preprocess))
#data, meta = loadarff('datasets/adult-test.arff')
#preprocess = MyPreprocessing(data)
#preprocess.fit()
#print(preprocess.new_df)