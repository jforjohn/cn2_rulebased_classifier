from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

class MyCN2(BaseEstimator, TransformerMixin):
    def __init__(self, beam_width=3, min_significance=0.5, negate=True):
        self.beam_width = beam_width
        self.min_significance = min_significance
        self.negate = negate
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
            for val in e[col].unique():
                selectors.append([(col, val, False)])
                if self.negate:
                    selectors.append([(col, val, True)])
        self.selectors = selectors


        #df_star = self.build_star()
        #star = df_star.iloc[:self.beam_width-1, :]
        #best_cpx = star['rule'].iloc[0]
        best_cpx = [[]]
        df_best = pd.DataFrame()
        #best_significance
        while not e.empty and best_cpx:
            df_best_cpx, best_cpx = self.find_best_complex(e, 
                                                           global_class_freqs,
                                                           df_shape)
            df_best = df_best.append(df_best_cpx)
            print('##################')
            print(df_best)
            print('##################')

            
            if best_cpx:
                # rule as dictionary key:attr, value: attr's value

                rule_filter = self.build_rule_filter(best_cpx, e)
                #complex_coverage = e[rule_filter]
                #e_prime = e[rule_filter]

                # remove from e the e_prime
                e = e[~rule_filter]
                print(e.shape)
        return self


    def find_best_complex(self, e, global_class_freqs, df_shape):
        # initial simple example of star
        star = [()]
        best_cpx = None
        best_significance = 1
        results = pd.DataFrame()
        while len(star) > 0 and best_significance > self.min_significance:
            new_star = list(filter(None, self.specialization(star)))
            print(len(new_star))
            rule_lst = []
            for cpx in new_star:
                rule_stats = {}
                
                rule_filter = self.build_rule_filter(cpx, e)
                
                complex_coverage = e[rule_filter]
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
                sort_order = ['entropy', 'significance', 'coverage']
                #sort_order = ['coverage', 'entropy', 'significance']
                asc_order = [True, False, False]
                df_best_cpxs = pd.DataFrame(rule_lst).sort_values(by=sort_order,
                        ascending=asc_order).iloc[:self.beam_width-1]

                results = results.append(df_best_cpxs)
                print(results)
                results = results.sort_values(by=sort_order,
                            ascending=asc_order).iloc[:self.beam_width]

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
                selector_attr, _, _ = selector[0]
            
                if star_cp:
                    if selector_attr not in [attr for attr, _, _ in star_cp]:
                        new_specialization = star_cp+[selector[0]]
                        for spec_item in specializations_lst:
                            if not set(spec_item).intersection(new_specialization):
                                specializations_lst.append(star_cp+[selector[0]])

                else:
                    # initial condition
                    specializations_lst = self.selectors
                    return specializations_lst
        return specializations_lst
            
    def build_rule_filter(self, cpx, e):
        rule_pos = {}
        rule_neg = {}
        for att, val, negate in cpx:
            if negate:
                rule_neg[att] = [val]
            else:
                rule_pos[att] = [val]

        if rule_pos:
            rule_filter_pos= e[rule_pos.keys()].isin(rule_pos).any(axis=1)
        else:
            rule_filter_pos = True
        
        if rule_neg:
            rule_filter_neg = ~e[rule_neg.keys()].isin(rule_neg).any(axis=1)
        else:
            rule_filter_neg = True

        return rule_filter_pos & rule_filter_neg
    

    def calc_significance(self, class_freq, complex_coverage_size, global_class_freqs, df_shape):
        fi = class_freq/complex_coverage_size
        ei = global_class_freqs/df_shape[0]
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
