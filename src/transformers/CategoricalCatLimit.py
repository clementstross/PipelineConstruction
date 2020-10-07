import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import _deprecate_positional_args

class CategoricalCatLimit(BaseEstimator, TransformerMixin):
    """Limiting the number of categorical values in a column
    """

    @_deprecate_positional_args
    def __init__(self, cat_num=None, other_value=None):
        """Limiting the number of categorical values in a column

        Args:
            cat_num (list[int], optional): List of the number of categories the ith column should be trimmed to. Defaults to None.
            other_value (list, optional): List of the value additional columns are mapped to. Defaults to None.

        Raises:
            ValueError: cat_num and other_value must be same length
        """        
        if cat_num is not None and other_value is not None:
            if len(cat_num) != len(other_value):
                raise ValueError("length of cat_num and other_value must be the same") 

        self.cat_num = cat_num
        self.other_value = other_value
        
        

    def fit(self, X, y=None):
        
        X_=X.copy()
        self.allowed_value = []

        # If input is null set to default
        if self.cat_num is None:
            self.cat_num = [10 for ii in range(0,X_.shape[1])]
        else:
            if X_.shape[1] != len(self.cat_num):
                raise ValueError("Number of columns in X must match the length of cat_num") 

        # If null start with empty list
        if self.other_value is None:
            self.other_value = []
        else:
            if X_.shape[1] != len(self.cat_num):
                raise ValueError("Number of columns in X must match the length of other_value")  

        for ii in range(0,X_.shape[1]):
            unique, counts = np.unique(X_.iloc[:,ii], return_counts=True)
            sorted_list = {k: v for k, v in sorted(dict(zip(unique, counts)).items(), key=lambda item: item[1], reverse=True)}

            self.allowed_value.append([*sorted_list][0:self.cat_num[ii]])
            
            if X_.iloc[:,ii].dtype in ['O','S'] and len(self.other_value)<=ii:
                self.other_value.append("other")
            elif X_.iloc[:,ii].dtype in ['I','F', "int32", "float32", "int64", "float64"] and len(self.other_value)<=ii:
                self.other_value.append(-1)
            elif len(self.other_value)<=ii:
                raise ValueError(f"Data type of {X_.iloc[:,ii].dtype} not know")

    def transform(self, X, y=None):
        X_=X.copy()

        if isinstance(X, np.ndarray):
            for col_ii in range(0,X_.shape[1]): 
                logic = [v not in self.allowed_value[col_ii] for v in X.iloc[:,col_ii].values]
                X_[logic, col_ii] = self.other_value[col_ii]
        
        elif isinstance(X, pd.DataFrame):

            for col_ii in range(0, X_.shape[1]):
                logic = [v not in self.allowed_value[col_ii] for v in X.iloc[:,col_ii].values]

                X_.iloc[logic, col_ii] = self.other_value[col_ii]

        return X_

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        X_ = self.transform(X, y)
        return X_
