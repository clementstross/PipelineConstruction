import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.validation import _deprecate_positional_args
#from sklearn.utils. import _unique, _check_unknown

#from pandas.core.common import flatten

# Built on top of OrdinalEncoder
# New code is labelled
class FrequencyEncoder(OrdinalEncoder):

    @_deprecate_positional_args
    def __init__(self, *, categories='auto', dtype=np.float64,
                 handle_unknown='error', unknown_value=None,
                 keep_at_start=None, keep_at_end=None):
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value

        # New code
        self.keep_at_start = keep_at_start
        self.keep_at_end = keep_at_end
        # End of new code

    def _fit(self, X, handle_unknown='error'):
        X_list, n_samples, n_features = self._check_X(X)

        self.categories_ = []

        if self.keep_at_start is None:
            self.keep_at_start = [[] for i in range(n_features)]
        elif len(self.keep_at_start) == 0:
            self.keep_at_start = [[] for i in range(n_features)]
        elif isinstance(self.keep_at_start[0], list) == False:
            self.keep_at_start = [[self.keep_at_start] for i in range(n_features)]
            
        if self.keep_at_end is None:
            self.keep_at_end = [[] for i in range(n_features)]
        elif len(self.keep_at_end) == 0:
            self.keep_at_end = [[] for i in range(n_features)]
        elif isinstance(self.keep_at_end[0], list) == False:
            self.keep_at_end = [[self.keep_at_end] for i in range(n_features)]


        for i in range(n_features):
            Xi = X_list[i]
            if self.categories == 'auto':
                ### New code
                unique, counts = np.unique(Xi, return_counts=True)

                flatten = lambda l: [item for sublist in l for item in sublist]

                itemindex = np.where(np.isin(unique, flatten([self.keep_at_start[i], self.keep_at_end[i]])) == True)
                
                #drop
                unique = np.delete(unique, itemindex)
                counts = np.delete(counts, itemindex)

                cats_dict = {k: v for k, v in sorted(dict(zip(unique, counts)).items(), key=lambda item: item[1], reverse=True)}
                cats = flatten([self.keep_at_start[i], [*cats_dict], self.keep_at_end[i]])
                
                ### End of New code

            else:
                cats = np.array(self.categories[i], dtype=Xi.dtype)
                if Xi.dtype != object:
                    if not np.all(np.sort(cats) == cats):
                        raise ValueError("Unsorted categories are not "
                                         "supported for numerical categories")
                if handle_unknown == 'error':
                    diff = _check_unknown(Xi, cats)
                    if diff:
                        msg = ("Found unknown categories {0} in column {1}"
                               " during fit".format(diff, i))
                        raise ValueError(msg)
            self.categories_.append(cats)


