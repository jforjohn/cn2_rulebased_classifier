from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


class MyCN2(BaseEstimator, TransformerMixin):
    def __init__(self, beam_width=3, min_significance=0.5, negate=False, disjunctive=False):
        self.beam_width = beam_width
        self.min_significance = min_significance
        self.negate = negate
        self.disjunctive = disjunctive
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.max_colwidth', 1000)
                
    def fit(self, dt):
        if isinstance(dt, pd.DataFrame):
            df = dt
        elif isinstance(dt, np.ndarray):
            df = pd.DataFrame(dt)
        else:
            raise Exception('dt should be a DataFrame or a numpy array')
        e = df.copy()
        global_class_freqs = df['Class'].value_counts()
        df_shape = df.shape
        selectors = []
        # don't use class column
        for col in e.columns[:-1]:
            for val in e.loc[:, col].unique():
                selectors.append([(col, val, False)])
                if self.negate:
                    selectors.append([(col, val, True)])
        self.selectors = selectors

        #df_star = self.build_star()
        #star = df_star.iloc[:self.beam_width-1, :]
        #best_cpx = star['rule'].iloc[0]
        best_cpx = [[]]
        df_rules = pd.DataFrame()
        #best_significance
        while not e.empty and best_cpx:
            df_best_cpx, best_cpx = self.find_best_complex(e, 
                                                           global_class_freqs,
                                                           df_shape)
            df_rules = df_rules.append(df_best_cpx)
            #print('##################')
            #print(df_rules)
            #print('##################')

            if best_cpx:
                # rule as dictionary key:attr, value: attr's value

                rule_filter = self.build_rule_filter(best_cpx, e)
                #print('oooooooooooooooooooo')
                #print(e[rule_filter].loc[:, 'Class'].value_counts())
                #print('oooooooooooooooooooo')
                #complex_coverage = e[rule_filter]
                #e_prime = e[rule_filter]

                # remove from e the e_prime
                e = e[~rule_filter]
                print(e.shape)
        
        rule_stats = {}
        rule_default = [('True')]
        rule_stats['rule'] = rule_default
        if e.shape[0] > 0:
            class_freq = e.loc[:, 'Class'].value_counts()
        else:
            class_freq = global_class_freqs
        rule_stats['entropy'] = entropy(class_freq, base=2)
        rule_stats['length'] = len(rule_default)
        rule_stats['coverage'] = e.shape[0]
        # the prediction is the most frequent class
        rule_stats['prediction'] = class_freq.index[0]
        # take the most frequent class for this rule
        rule_stats['precision'] = class_freq[0]/class_freq.sum()
        most_freq_class_in_dataset = global_class_freqs.loc[class_freq.index[0]]
        rule_stats['recall'] = class_freq[0] / most_freq_class_in_dataset
        rule_stats['significance'] = self.calc_significance(class_freq, 
                                                            e.shape[0],
                                                            global_class_freqs,
                                                            df_shape)

        self.df_rules = df_rules.append(pd.DataFrame([rule_stats]))
        self.df_rules.reset_index(drop=True, inplace=True)
        print(self.df_rules)
        print('negate', self.negate)
        print('disjunctive', self.disjunctive)
        print()
        print('#####df_rules#####')
        print(df_rules.values.tolist())
        print('##################')
        print()
        return self


    def find_best_complex(self, e, global_class_freqs, df_shape):
        # initial simple example of star
        star = [()]
        best_cpx = None
        best_significance = 1
        results = pd.DataFrame()
        while len(star) > 0 and best_significance > self.min_significance:
            new_star = list(filter(None, self.specialization(star)))
            '''
            print('-----------------')
            print('len star sent', len(star))
            print('star sent')
            print(star)
            print('---------             --------')
            print('new star produced')
            print(new_star)
            print('-----------------')
            print('len new_star', len(new_star))
            '''
            rule_lst = []
            for cpx in new_star:
                rule_stats = {}
                
                rule_filter = self.build_rule_filter(cpx, e)
                
                complex_coverage = e[rule_filter]
                # stats
                if complex_coverage.size > 0:
                    rule_stats['rule'] = cpx
                    class_freq = complex_coverage.loc[:, 'Class'].value_counts()
                    rule_stats['entropy'] = entropy(class_freq, base=2)
                    rule_stats['length'] = len(cpx)
                    rule_stats['coverage'] = complex_coverage.shape[0]
                    # the prediction is the most frequent class
                    rule_stats['prediction'] = class_freq.index[0]
                    # take the most frequent class for this rule
                    rule_stats['precision'] = class_freq[0]/class_freq.sum()
                    most_freq_class_in_dataset = global_class_freqs.loc[class_freq.index[0]]
                    rule_stats['recall'] = class_freq[0] / most_freq_class_in_dataset
                    rule_stats['significance'] = self.calc_significance(class_freq, 
                                                                        complex_coverage.shape[0],
                                                                        global_class_freqs,
                                                                        df_shape)
                    rule_lst.append(rule_stats)
                #else:
                #    del new_star[ind]
            if rule_lst:
                sort_order = ['entropy', 'significance', 'length', 'coverage']
                #sort_order = ['coverage', 'entropy', 'significance']
                asc_order = [True, False, True, False]
                df_best_cpxs = pd.DataFrame(rule_lst).sort_values(by=sort_order,
                        ascending=asc_order).iloc[:self.beam_width-1]

                #print('df_best_cpxs', df_best_cpxs)
                results = results.append(df_best_cpxs)
                #print('results', results)
                results = results.sort_values(by=sort_order,
                            ascending=asc_order).iloc[:self.beam_width]

                best_cpx = results['rule'].iloc[0]
                #best_significance = results['significance'].iloc[0]
                best_significance = df_best_cpxs['significance'].iloc[0]
                star = df_best_cpxs['rule'].values.tolist()
            else:
                star = []
        return results.iloc[0,:], best_cpx


    def specialization(self, star):
        specializations_lst = [[]]
        for s_item in star:
            star_cp = s_item#.copy()
            for selector in self.selectors:
                selector_attr, _, _ = selector[0]
            
                if star_cp:
                    if selector_attr not in [attr for attr, _, _ in star_cp]:
                        new_specialization = star_cp+[selector[0]]
                        ind = 0
                        foundSame = False
                        #print('specializations_lst')
                        #print(specializations_lst)
                        while ind < len(specializations_lst) and not foundSame:
                            spec_item = specializations_lst[ind]
                            new_spec_intersection = set(spec_item).intersection(new_specialization)
                            if len(new_spec_intersection) == len(new_specialization):
                                foundSame = True
                            ind += 1
                        if not foundSame:
                            specializations_lst.append(new_specialization)
                else:
                    # initial condition
                    return self.selectors
        return specializations_lst

    def predict(self, dt, y):
        if isinstance(dt, pd.DataFrame):
            df_test = dt
        elif isinstance(dt, np.ndarray):
            df_test = pd.DataFrame(dt)
        else:
            raise Exception('dt should be a DataFrame or a numpy array')
        
        if isinstance(y, pd.DataFrame):
            y_test = y.copy()
        elif isinstance(dt, np.ndarray):
            y_test = pd.DataFrame(y.copy())
        else:
            raise Exception('y should be a DataFrame or a numpy array')
        
        if not hasattr(self, 'df_rules'):
            raise Exception('The model has to be fitted first')

        y_test.loc[:, 'Predictions'] = np.nan

        global_class_freqs = df_test.loc[:, 'Class'].value_counts()
        df_shape = df_test.shape
        df_results = pd.DataFrame()
        rule_lst = []
        # don't iterate through the default rule
        last_rule_ind = self.df_rules.shape[0]-2
        for ind, rule in enumerate(self.df_rules.loc[:last_rule_ind, 'rule']):
            print('test shape', df_test.shape)
            rule_stats = {}
            
            rule_filter = self.build_rule_filter(rule, df_test)

            complex_coverage = df_test[rule_filter]
            #complex_coverage_labels = y_test[rule_filter]
            
            # stats

            if complex_coverage.size > 0:
                rule_stats['rule'] = rule
                # the prediction is the most frequent class
                prediction = self.df_rules.loc[ind, 'prediction']
                rule_stats['prediction'] = prediction
                y_test.loc[complex_coverage.index, 'Predictions'] = prediction
                # what's the frequency of 
                class_freq = complex_coverage[complex_coverage.loc[:,'Class']==prediction].loc[:, 'Class'].value_counts()
                rule_stats['entropy'] = entropy(class_freq, base=2)
                rule_stats['length'] = len(rule)
                rule_stats['coverage'] = complex_coverage.shape[0]

                rule_stats['train_precision'] = self.df_rules.loc[ind, 'precision']
                # take the most frequent class for this rule
                rule_stats['true_most_freq_value'] = complex_coverage.loc[:,'Class'].value_counts().index[0]
                if class_freq.shape[0] > 0:
                    rule_stats['correct'] = class_freq[0]
                    rule_stats['accuracy'] = class_freq[0]/complex_coverage.shape[0]
                    most_freq_class_in_dataset = global_class_freqs.loc[class_freq.index[0]]
                    rule_stats['recall'] = class_freq[0] / most_freq_class_in_dataset
                    rule_stats['significance'] = self.calc_significance(class_freq, 
                                                                complex_coverage.shape[0],
                                                                global_class_freqs,
                                                                df_shape)
                else:
                    rule_stats['correct'] = 0
                    rule_stats['accuracy'] = 0
                    rule_stats['recall'] = 0
                    rule_stats['significance'] = 0
                    
                rule_lst.append(rule_stats)
            df_test = df_test[~rule_filter]
        if df_test.shape[0] > 0:
            print('DEFAULT')
            complex_coverage = df_test
            ind = self.df_rules.shape[0]-2
            rule = self.df_rules.loc[ind, 'rule']
            prediction = self.df_rules.loc[ind, 'prediction']
            rule_stats['prediction'] = prediction
            y_test.loc[complex_coverage.index, 'Predictions'] = prediction
            # what's the frequency of 
            class_freq = complex_coverage[complex_coverage.loc[:,'Class']==prediction].loc[:, 'Class'].value_counts()
            rule_stats['entropy'] = entropy(class_freq, base=2)
            rule_stats['length'] = len(rule)
            rule_stats['coverage'] = complex_coverage.shape[0]

            rule_stats['train_precision'] = self.df_rules.loc[ind, 'precision']
                # take the most frequent class for this rule
            rule_stats['true_most_freq_value'] = complex_coverage.loc[:,'Class'].value_counts().index[0]
            rule_stats['correct'] = class_freq[0]
            rule_stats['accuracy'] = class_freq[0]/complex_coverage.shape[0]
            most_freq_class_in_dataset = global_class_freqs.loc[class_freq.index[0]]
            rule_stats['recall'] = class_freq[0] / most_freq_class_in_dataset
            rule_stats['significance'] = self.calc_significance(class_freq, 
                                                            complex_coverage.shape[0],
                                                            global_class_freqs,
                                                            df_shape)
            rule_lst.append(rule_stats)

        df_results = pd.DataFrame(rule_lst)
        print(df_results)
        print()
        print('#####df_results#####')
        print(df_results.values.tolist())
        print('##################')
        print()
        return y_test.loc[:, 'Predictions']

    def calc_significance(self, class_freq, complex_coverage_size, global_class_freqs, df_shape):
        fi = class_freq/complex_coverage_size
        ei = global_class_freqs/df_shape[0]
        return 2*(fi*np.log(fi/ei)).sum()

    def build_rule_filter(self, cpx, e):
        rule_pos = {}
        rule_neg = {}
        for att, val, negate in cpx:
            if negate:
                rule_neg[att] = [val]
            else:
                rule_pos[att] = [val]

        if rule_pos:
            if self.disjunctive:
                rule_filter_pos= e[rule_pos.keys()].isin(rule_pos).any(axis=1)
            else:
                rule_filter_pos= e[rule_pos.keys()].isin(rule_pos).all(axis=1)
        else:
            if self.disjunctive:
                rule_filter_pos = False
            else:
                rule_filter_pos = True
        
        if rule_neg:
            if self.disjunctive:
                rule_filter_neg = ~e[rule_neg.keys()].isin(rule_neg).any(axis=1)
            else:
                rule_filter_neg = ~e[rule_neg.keys()].isin(rule_neg).all(axis=1)
        else:
            if self.disjunctive:
                rule_filter_neg = False
            else:
                rule_filter_neg = True

        if self.disjunctive:
            rule_filter = rule_filter_pos | rule_filter_neg
        else:
            rule_filter = rule_filter_pos & rule_filter_neg
        return rule_filter 