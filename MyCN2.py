from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

class MyCN2(BaseEstimator, TransformerMixin):
    def __init__(self, beam_width=3, min_significance=0.5):
        self.beam_width = beam_width
        self.min_significance = min_significance
                
    def fit(self, dt):
        if isinstance(dt, pd.DataFrame):
            self.e = dt
        elif isinstance(dt, np.ndarray):
            self.e = pd.DataFrame(dt)
        else:
            raise Exception('dt should be a DataFrame or a numpy array')

        selectors = []
        # don't use class column
        for col in self.e.columns[:-1]:
            selectors.extend([[(col, val)] for val in self.e[col].unique()])
        self.selectors = selectors


        #df_star = self.build_star()
        #star = df_star.iloc[:self.beam_width-1, :]
        #best_cpx = star['rule'].iloc[0]
        best_cpx = [[]]
        df_best = pd.DataFrame()
        #best_significance
        while not self.e.empty and best_cpx:
            print('edo')
            df_best_cpx, best_cpx = self.find_best_complex()
            df_best = df_best.append(df_best_cpx)
            print('##################')
            print(df_best)
            print('##################')

            
            if best_cpx:
                # rule as dictionary key:attr, value: attr's value
                rule_cpx = {}
                for att, val in best_cpx:
                    rule_cpx[att] = [val]

                rule_filter = self.e[rule_cpx.keys()].isin(rule_cpx).any(axis=1)
                #complex_coverage = self.e[rule_filter]
                #e_prime = self.e[rule_filter]

                # remove from e the e_prime
                self.e = self.e[~rule_filter]
                print(self.e.shape)


    def find_best_complex(self):
        print('pou')
        # initial simple example of star
        star = [()]
        best_cpx = None
        best_significance = 1
        results = pd.DataFrame()
        while len(star) > 0 and best_significance > self.min_significance:
            new_star = list(filter(None, self.specialization(star)))
            print(len(new_star))
            rule_lst = []
            for ind, cpx in enumerate(new_star):
                rule = {}
                rule_stats = {}
                for att, val in cpx:
                    rule[att] = [val]

                rule_filter = self.e[rule.keys()].isin(rule).any(axis=1)
                complex_coverage = self.e[rule_filter]
                # stats
                if complex_coverage.size > 0:
                    rule_stats['rule'] = cpx
                    class_freq = complex_coverage['Class'].value_counts()
                    rule_stats['entropy'] = entropy(class_freq, base=2)
                    rule_stats['length'] = len(cpx)
                    rule_stats['coverage'] = complex_coverage.shape[0]
                    # the prediction is the most frequent class
                    rule_stats['prediction'] = class_freq.index[0]
                    # take the most frequent class for this rule
                    rule_stats['precision'] = class_freq[0]/class_freq.sum()
                    most_freq_class_in_dataset = self.e.Class.value_counts().loc[class_freq.index[0]]
                    rule_stats['recall'] = class_freq[0] / most_freq_class_in_dataset
                    rule_stats['significance'] = self.calc_significance(complex_coverage)
                    rule_lst.append(rule_stats)
                #else:
                #    del new_star[ind]
            if rule_lst:
                sort_order = ['entropy', 'significance', 'coverage']
                #sort_order = ['coverage', 'entropy', 'significance']
                asc_order = [True, False, False]
                df_best_cpxs = pd.DataFrame(rule_lst).sort_values(by=sort_order,
                        ascending=asc_order).iloc[:self.beam_width-1]

                results = results.append(df_best_cpxs)
                results = results.sort_values(by=sort_order,
                            ascending=asc_order).iloc[:self.beam_width]
                print(results)

                best_cpx = results['rule'].iloc[0]
                best_significance = results['significance'].iloc[0]
                star = df_best_cpxs['rule'].values.tolist()
            else:
                star = []
        return results.iloc[0,:], best_cpx


    def specialization(self, star):
        specializations_lst = [[]]
        for s_item in star:
            star_cp = s_item#.copy()
            for selector in self.selectors:
                selector_attr, _ = selector[0]
            
                if star_cp:
                    if selector_attr not in [attr for attr, _ in star_cp]:
                        new_specialization = star_cp+[selector[0]]
                        for spec_item in specializations_lst:
                            if not set(spec_item).intersection(new_specialization):
                                specializations_lst.append(star_cp+[selector[0]])

                else:
                    # initial condition
                    specializations_lst = self.selectors
                    return specializations_lst
        return specializations_lst
            

            
    def build_star(self):
        rule_lst = []
        for selector in self.selectors:
            attr = [a for a,_ in selector]

            rule_stats = {}
            rule_stats['rule'] = selector
            # null complex
            if len(set(attr)) < len(attr):
                #rule = None
                #complex_coverage = pd.DataFrame()
                rule_stats['length'] = 0
                rule_stats['coverage'] = 0
                rule_stats['precision'] = 0
                rule_stats['recall'] = 0
                rule_stats['prediction'] = None
                rule_stats['entropy'] = 42
                rule_stats['significance'] = 0
            else:
                rule = {}
                for att, val in selector:
                    rule[att] = [val]

                rule_filter = self.e[rule.keys()].isin(rule).any(axis=1)
                complex_coverage = self.e[rule_filter]
                # stats
                rule_stats['length'] = len(selector)
                rule_stats['coverage'] = complex_coverage.shape[0]
                # take the most frequent class for this rule
                class_freq = complex_coverage['Class'].value_counts()
                rule_stats['precision'] = class_freq[0]/class_freq.sum()
                most_freq_class_in_dataset = self.e.Class.value_counts().loc[class_freq.index[0]]
                rule_stats['recall'] = class_freq[0] / most_freq_class_in_dataset
                # the prediction is the most frequent class
                rule_stats['prediction'] = class_freq.index[0]
                rule_stats['entropy'] = entropy(class_freq, base=2)
                rule_stats['significance'] = self.calc_significance(complex_coverage)
            
            rule_lst.append(rule_stats)
        df_rules = pd.DataFrame(rule_lst).sort_values(by=['entropy', 'significance', 'coverage'],
                    ascending=[True, False, False])
        return df_rules



    def calc_significance(self, complex_coverage):
        class_freq = complex_coverage['Class'].value_counts()
        fi = class_freq/complex_coverage.shape[0]
        global_class_stats = self.e['Class'].value_counts()
        ei = global_class_stats/self.e.shape[0]
        return 2*(fi*np.log(fi/ei)).sum()


'''
data = np.array([[2,3],
                 [3,5],
                 [1,4],
                 [10,12],
                 [11,13],
                 [12,10]])

plt.scatter(data[:,0], data[:,1], s=100)
#plt.show()
df = pd.DataFrame(data)

clf = MyCN2()

'''
