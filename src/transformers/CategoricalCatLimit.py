import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import _deprecate_positional_args

from sklearn.preprocessing import OrdinalEncoder

class CategoricalCatLimit(OrdinalEncoder):
    """Limiting the number of categorical values in a column
    """

    @_deprecate_positional_args
    def __init__(self, cat_num=None, other_value="auto"):
        """Limiting the number of categorical values in a column

        Args:
            cat_num (list[int], optional): List of the number of categories the ith column should be trimmed to. Defaults to None.
            other_value (list, optional): List of the value additional columns are mapped to. Defaults to None. Meaning string "Other" is used

        Raises:
            ValueError: cat_num and other_value must be same length
        """        

        self.cat_num = cat_num
        self.cat_num_ = []
        self.other_value = other_value

        if cat_num==cat_num and other_value!="auto":
            if len(cat_num)!=len(other_value):
                raise ValueError("If other_value is not 'auto' and cat_num is filled cat_num and other_value must be of the same length")
        
        

    def fit(self, X, y=None):
        
        X_list, n_samples, n_features = self._check_X(X)
        self.allowed_value = []

        # If input cat_num is null set to default which is 10
        if self.cat_num is None:
            self.cat_num_ = [10 for ii in range(n_features)]
        else:
            self.cat_num_ = self.cat_num.copy()
            if len(self.cat_num_) != n_features:
                raise ValueError("Shape mismatch: if cat_num is an list,"
                                 " it has to be of length (n_features).")
            for ii in self.cat_num_:
                if (int(ii)!=ii and ii>1) or ii<0:
                    raise ValueError("values in self.cat_num must be positive integer or in range [0-1]")

        # If input other_value null start with empty list
        if self.other_value == "auto":
            self.other_value_ = []
        else:
            self.other_value_=self.other_value
            if n_features != len(self.other_value_):
                raise ValueError("Number of columns in X must match the length of other_value")  
        
        for ii in range(0,n_features):
            Xi = X_list[ii]
            unique, counts = np.unique(Xi, return_counts=True)
            sorted_dict = {k: v for k, v in sorted(dict(zip(unique, counts)).items(), key=lambda item: item[1], reverse=True)}

            #If value is integer Cap on number of levels
            if self.cat_num_[ii]>=1:
                self.allowed_value.append([*sorted_dict][0:self.cat_num_[ii]])
            elif self.cat_num_[ii]>0 and self.cat_num_[ii]<1:
                cat_num_calc = max([ii if vv/sum(counts) else 0 for ii, (kk, vv) in enumerate(sorted_dict.items())])
                self.cat_num_[ii] = cat_num_calc
                self.allowed_value.append([*sorted_dict][0:cat_num_calc])
            else:
                raise ValueError("values in self.cat_num must be positive integer or in range [0-1]")

            if isinstance(Xi[0],str) and len(self.other_value_)<=ii:
                self.other_value_.append("other")
            elif isinstance(Xi[0],(int, float, np.int, np.int32, np.int64, np.float, np.float32, np.float64)) and len(self.other_value_)<=ii:
                self.other_value_.append(-1)
            elif len(self.other_value_)<=ii:
                raise ValueError(f"Data type of {type(Xi[0])} not know")


    def transform(self, X, y=None):
        X_list, n_samples, n_features = self._check_X(X)

        if n_features != len(self.allowed_value):
            raise ValueError(
                "The number of features in X is different to the number of "
                "features of the fitted data. The fitted data had {} features "
                "and the X has {} features."
                .format(len(self.allowed_value,), n_features)
            )


        X_ = X_list.copy()
        for ii in range(n_features): 
            X_[ii] = [v if v in self.allowed_value[ii] else self.other_value_[ii] for v in X_list[ii]]

        return X_

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        X_ = self.transform(X, y)
        return X_

    def inverse_transform(self, X):
        return X.copy()
