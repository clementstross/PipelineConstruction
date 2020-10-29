import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import FLOAT_DTYPES, _deprecate_positional_args

class NumericCapping(TransformerMixin, BaseEstimator):
    
    @_deprecate_positional_args
    def __init__(self, unk_max=None, cap_max=None, cap_min=None, unk_min=None, copy=True):

        # Error checking on order when values aren't NA
        if unk_max < cap_max and unk_max==unk_max and cap_max==cap_max:
            raise ValueError(f"unk_max ({unk_max}) must be >= cap_max ({cap_max})")
        if unk_max <= cap_min and unk_max==unk_max and cap_min==cap_min:
            raise ValueError(f"unk_max ({unk_max}) must be > cap_min ({cap_min})")
        if unk_max <= unk_min and unk_max==unk_max and unk_min==unk_min:
            raise ValueError(f"unk_max ({unk_max}) must be > unk_min ({unk_min})")

        if cap_max <= cap_min and cap_max==cap_max and cap_min==cap_min:
            raise ValueError(f"cap_max ({cap_max}) must be > cap_min ({cap_min})")
        if cap_max <= unk_min and cap_max==cap_max and unk_min==unk_min:
            raise ValueError(f"cap_max ({cap_max}) must be > unk_min ({unk_min})")

        if cap_min < unk_min and cap_min==cap_min and unk_min==unk_min:
            raise ValueError(f"cap_min ({cap_min}) must be >= unk_min ({unk_min})")

        self.unk_max=unk_max
        self.cap_max=cap_max
        self.cap_min=cap_min
        self.unk_min=unk_min

        self.copy = copy


    def _reset(self):
        """Reset internal data-dependent state of the transformer, if necessary.
        __init__ parameters are not touched.
        """

        # Checking one attribute is enough, becase they are all set together
        # in partial_fit
        if hasattr(self, 'n_samples_seen_'):
            del self.n_samples_seen_


    def fit(self, X, y=None, sample_weight=None):
        """Compute the mean and std to be used for later scaling.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to compute the features axis.
        y : None
            Ignored.
        sample_weight : array-like of shape (n_samples,), default=None
            Ignored.
        Returns
        -------
        self : object
            Fitted scaler.
        """

        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X, y, sample_weight)

    def partial_fit(self, X, y=None, sample_weight=None):
        first_call = not hasattr(self, "n_samples_seen_")
        X = self._validate_data(X, accept_sparse=('csr', 'csc'),
                                estimator=self, dtype=FLOAT_DTYPES,
                                force_all_finite='allow-nan', reset=first_call)

        if self.n_features_in_ != len(self.unk_max):
            raise ValueError("Number of columns in X must match the length of unk_max")
        if self.n_features_in_ != len(self.cap_max):
            raise ValueError("Number of columns in X must match the length of cap_max")
        if self.n_features_in_ != len(self.cap_min):
            raise ValueError("Number of columns in X must match the length of cap_min")
        if self.n_features_in_ != len(self.unk_min):
            raise ValueError("Number of columns in X must match the length of unk_min") 

        return self

    def transform(self, X, y=None, copy=None):
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
        copy = copy if copy is not None else self.copy
        X = self._validate_data(X, reset=False,
                                accept_sparse='csr', copy=copy,
                                estimator=self, dtype=FLOAT_DTYPES,
                                force_all_finite='allow-nan')
        X_ = X.copy()
        X2_ = X_.copy()

        # For each column of the data and each unknown value in that colum replace levels with unknown
        for col_ii in range(0, X_.shape[1]):
            # Replace with correct type of unknown
            X_[X2_[:, col_ii]  > self.cap_max[col_ii], col_ii] = self.cap_max[col_ii]
            X_[X2_[:, col_ii]  < self.cap_min[col_ii], col_ii] = self.cap_min[col_ii]
            
            if X_[:, col_ii].dtype == "float64":
                X_[X2_[:, col_ii]  > self.unk_max[col_ii], col_ii] = np.nan
                X_[X2_[:, col_ii]  < self.unk_min[col_ii], col_ii] = np.nan
            elif X_[:, col_ii].dtype == "int":
                X_[X2_[:, col_ii]  > self.unk_max[col_ii], col_ii] = -999999
                X_[X2_[:, col_ii]  < self.unk_min[col_ii], col_ii] = -999999
                
               
                    
        return X_

if __name__=="__main__":
    num_cap = NumericCapping(unk_max=[9], cap_max=[8], cap_min=[3], unk_min=[2])
    X=np.array([[1.], [2.], [3.], [4.], [5.], [6.], [7.], [8.], [9.], [10.]])
    X_out = num_cap.fit_transform(X=X)
    print(X_out)

    X_df=pd.DataFrame(X)
    X_df_out = num_cap.fit_transform(X=X_df)
    print(X_df_out)


    num_cap2 = NumericCapping(unk_max=[9, 90], cap_max=[8, 80], cap_min=[3, 30], unk_min=[2, 20])
    X2=np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50], [6, 60], [7, 70], [8, 80], [9, 90], [10, 100]])
    X2_out = num_cap2.fit_transform(X=X2)
    print(X2_out)

    X2_df=pd.DataFrame(X2)
    X2_df_out = num_cap2.fit_transform(X=X2_df)
    print(X2_df_out)