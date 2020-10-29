import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.validation import _deprecate_positional_args
#from sklearn.utils import _check_unknown

#from pandas.core.common import flatten

# Built on top of OrdinalEncoder
# New code is labelled
class FrequencyEncoder(OrdinalEncoder):

    @_deprecate_positional_args
    def __init__(self, *, categories='auto', dtype=np.float64,
                 handle_unknown='error', unknown_value=None,
                 keep_at_start=None, keep_at_end=None):
        self.categories = categories
        self.categories_ = []
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
                cats = np.array(flatten([self.keep_at_start[i], [*cats_dict], self.keep_at_end[i]]), dtype='object')
                
                #The transformation code doesn't work with numeric values
                if Xi.dtype != object :
                    if not np.all(np.sort(cats) == cats):
                        raise ValueError("Unsorted categories are not "
                                         "supported for numerical categories")

                ### End of New code

            else:
                cats = np.array(self.categories[i], dtype=Xi.dtype)
                if Xi.dtype != object:
                    if not np.all(np.sort(cats) == cats):
                        raise ValueError("Unsorted categories are not "
                                         "supported for numerical categories")
                if handle_unknown == 'error':
                    #diff = _check_unknown(Xi, cats)
                    diff = False
                    if diff:
                        msg = ("Found unknown categories {0} in column {1}"
                               " during fit".format(diff, i))
                        raise ValueError(msg)

            self.categories_.append(cats)


### copied from sklearn utils
def _check_unknown(values, known_values, return_mask=False):
    """
    Helper function to check for unknowns in values to be encoded.
    Uses pure python method for object dtype, and numpy method for
    all other dtypes.
    Parameters
    ----------
    values : array
        Values to check for unknowns.
    known_values : array
        Known values. Must be unique.
    return_mask : bool, default=False
        If True, return a mask of the same shape as `values` indicating
        the valid values.
    Returns
    -------
    diff : list
        The unique values present in `values` and not in `know_values`.
    valid_mask : boolean array
        Additionally returned if ``return_mask=True``.
    """
    valid_mask = None

    if values.dtype.kind in 'UO':
        values_set = set(values)
        values_set, missing_in_values = _extract_missing(values_set)

        uniques_set = set(known_values)
        uniques_set, missing_in_uniques = _extract_missing(uniques_set)
        diff = values_set - uniques_set

        nan_in_diff = missing_in_values.nan and not missing_in_uniques.nan
        none_in_diff = missing_in_values.none and not missing_in_uniques.none

        def is_valid(value):
            return (value in uniques_set or
                    missing_in_uniques.none and value is None or
                    missing_in_uniques.nan and is_scalar_nan(value))

        if return_mask:
            if diff or nan_in_diff or none_in_diff:
                valid_mask = np.array([is_valid(value) for value in values])
            else:
                valid_mask = np.ones(len(values), dtype=bool)

        diff = list(diff)
        if none_in_diff:
            diff.append(None)
        if nan_in_diff:
            diff.append(np.nan)
    else:
        unique_values = np.unique(values)
        diff = np.setdiff1d(unique_values, known_values,
                            assume_unique=True)
        if return_mask:
            if diff.size:
                valid_mask = np.in1d(values, known_values)
            else:
                valid_mask = np.ones(len(values), dtype=bool)

        # check for nans in the known_values
        if np.isnan(known_values).any():
            diff_is_nan = np.isnan(diff)
            if diff_is_nan.any():
                # removes nan from valid_mask
                if diff.size and return_mask:
                    is_nan = np.isnan(values)
                    valid_mask[is_nan] = 1

                # remove nan from diff
                diff = diff[~diff_is_nan]
        diff = list(diff)

    if return_mask:
        return diff, valid_mask
    return diff