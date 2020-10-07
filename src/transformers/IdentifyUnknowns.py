import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import _deprecate_positional_args

class IdentifyUnknowns(BaseEstimator, TransformerMixin):
    """Removes known unknown values and replaces them with unknown values

    The unknowns values are values such as "NA", -1 and 999 which need mapping to np.nan for imputation.

    Parameters
    ----------
    unk_levels: nested list of length n_features
        contains values to be mapped to unknown

    Attributes
    ----------
    unk_levels: nested of length n_features
        each value of unk_levels is the unknown levels for that column

    Examples
    --------
    >>> from sklearn.base import BaseEstimator, TransformerMixin
    >>> from sklearn.utils.validation import _deprecate_positional_args

    >>> id_unk = IdentifyUnknowns(unk_levels=[[1.,2.], [3., 4.]])
    >>> X=np.array([[1., 1.], [2., 2,], [3., 3.], [4., 4.]])
    >>> X_out = id_unk.fit_transform(X=X2)
    >>> print(X2_out)
    [[nan  1.]
    [nan  2.]
    [ 3. nan]
    [ 4. nan]]
    """
    @_deprecate_positional_args
    def __init__(self, unk_levels=None):
        self.unk_levels = unk_levels

    def fit(self, X, y=None):
        """No fitting needed

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted scaler.
        """
        return self

    def transform(self, X, y=None):
        """Remove known unknowns from data.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed.
        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        X_ = X.copy()

        if isinstance(X, np.ndarray):
            # For each column of the data and each unknown value in that colum replace levels with unknown
            for col_ii in range(0, X_.shape[1]):
                for unk_level in self.unk_levels[col_ii]:
                    # Replace with correct type of unknown
                    if X_[:, col_ii].dtype == "float64":
                        X_[X_[:, col_ii]  == unk_level, col_ii] = np.nan
                    if X_[:, col_ii].dtype == "int":
                        X_[X_[:, col_ii]  == unk_level, col_ii] = -999999
                    if X_[:, col_ii].dtype == "str":
                        X_[X_[:, col_ii]  == unk_level, col_ii] = ""
        elif isinstance(X, pd.DataFrame):

            if X_.shape[1] != len(self.unk_levels):
                raise ValueError("Number of columns in X must match the length of unk_levels") 
            
            for col_ii in range(0, X_.shape[1]):
                for unk_level in self.unk_levels[col_ii]:
                    X_.iloc[X.iloc[:,col_ii].values == unk_level, col_ii] = np.nan
                    
        return X_

if __name__=="__main__":
    pass

    #Test numpy 
    id_unk = IdentifyUnknowns(unk_levels=[[1.,2.]])
    X=np.array([[1.], [2.], [3.], [4.]])
    X_out = id_unk.fit_transform(X=X)
    print(X_out)

    #Test pandas
    id_unk_df = IdentifyUnknowns(unk_levels=[[1.,2.]])
    X_df = pd.DataFrame(X, columns=["col1"])
    X_df_out = id_unk.fit_transform(X=X_df)
    print(X_df_out)

    #Test numpy
    id_unk2 = IdentifyUnknowns(unk_levels=[[1.,2.], [3., 4.]])
    X2=np.array([[1., 1.], [2., 2,], [3., 3.], [4., 4.]])
    X2_out = id_unk2.fit_transform(X=X2)
    print(X2_out)

    #Test pandas
    id_unk2_df = IdentifyUnknowns(unk_levels=[[1.,2.], [3., 4.]])
    X2_df = pd.DataFrame(X2, columns=["col1", "col2"])
    X2_df_out = id_unk2.fit_transform(X=X2_df)
    print(X2_df_out)

