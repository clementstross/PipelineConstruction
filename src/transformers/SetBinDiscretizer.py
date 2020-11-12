import numpy as np

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder

from sklearn.utils.validation import _deprecate_positional_args


class SetBinDiscretizer(KBinsDiscretizer):
    """
    Bin continuous data into user defined intervals.
    
    Parameters
    ----------
    bin_edges_internal : 2D list of shape (n_features, n_bins - 1)
        These are the internal bin boundaries. The lower boundary of each bin is inclusive
    encode : {'onehot', 'onehot-dense', 'ordinal'}, (default='onehot')
        Method used to encode the transformed result.
        onehot
            Encode the transformed result with one-hot encoding
            and return a sparse matrix. Ignored features are always
            stacked to the right.
        onehot-dense
            Encode the transformed result with one-hot encoding
            and return a dense array. Ignored features are always
            stacked to the right.
        ordinal
            Return the bin identifier encoded as an integer value.
    Attributes
    ----------
    n_bins_ : int array, shape (n_features,)
        Number of bins per feature. Bins whose width are too small
        (i.e., <= 1e-8) are removed with a warning.
    bin_edges_ : array of arrays, shape (n_features, )
        The edges of each bin. Contain arrays of varying shapes ``(n_bins_, )``
        Ignored features will have empty arrays.
    Notes
    -----
    Build on KBinsDiscretizer

    In bin edges for feature ``i``, the first and last values are used only for
    ``inverse_transform``. During transform, bin edges are extended to::
      np.concatenate([-np.inf, bin_edges_[i][1:-1], np.inf])
    You can combine ``KBinsDiscretizer`` with
    :class:`sklearn.compose.ColumnTransformer` if you only want to preprocess
    part of the features.
    ``KBinsDiscretizer`` might produce constant features (e.g., when
    ``encode = 'onehot'`` and certain bins do not contain any data).
    These features can be removed with feature selection algorithms
    (e.g., :class:`sklearn.feature_selection.VarianceThreshold`).
    Examples
    --------
    >>> X = [[-2, 1, -4, -1  ],
     ...     [-1, 2, -3, -0.5], 
     ...     [ 0, 3, -2,  0.5], 
     ...     [ 1, 4, -1,    2]]
    >>> sbd = SetBinDiscretizer(bin_edges_internal=[[-1, 0, 1], [1], [-3, -2], [-1, 0, 1]], encode='ordinal')
    >>> sbd.fit(X)
    SetBinDiscretizer(...)
    >>> Xt = sbd.transform(X)
    >>> Xt  # doctest: +SKIP
    array([[0., 0., 0., 0.],
          [1., 0., 1., 0.],
          [2., 0., 2., 1.],
          [2., 0., 2., 2.]])
    Sometimes it may be useful to convert the data back into the original
    feature space. The ``inverse_transform`` function converts the binned
    data into the original feature space. Each value will be equal to the mean
    of the two bin edges.
    >>> est.bin_edges_[0]
    array([-2., -1.,  0.,  1.])
    >>> est.inverse_transform(Xt)
    array([[-1.5,  2.5, -3.5, -0.5],
           [-0.5,  2.5, -2.5, -0.5],
           [ 0.5,  2.5, -1.5,  0.5],
           [ 0.5,  2.5, -1.5,  1.5]])
    """
    @_deprecate_positional_args
    def __init__(self, bin_edges_internal, *, encode='onehot', input_features=None):
        # bin_edges_internal must be 2D list.
        # Ensure values are unique and sorted
        self.bin_edges_internal = [sorted(list(set(bin_edges_internal_ii))) for bin_edges_internal_ii in bin_edges_internal]
        self.encode = encode

        self.input_features=input_features

    def fit(self, X, y=None):
        """"
        Fit the estimator.
        Parameters
        ----------
        X : numeric array-like, shape (n_samples, n_features)
            Data to be discretized.
        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`sklearn.pipeline.Pipeline`.
        Returns
        -------
        self
        """
        X = self._validate_data(X, dtype='numeric')

        valid_encode = ('onehot', 'onehot-dense', 'ordinal')
        if self.encode not in valid_encode:
            raise ValueError("Valid options for 'encode' are {}. "
                             "Got encode={!r} instead."
                             .format(valid_encode, self.encode))

        n_features = X.shape[1]

        bin_edges = np.zeros(n_features, dtype=object)
        for jj in range(n_features):
            column = X[:, jj]
            col_min, col_max = column.min(), column.max()

            #Add limits if needed
            bin_edges_temp = self.bin_edges_internal[jj]
            if col_min < min(bin_edges_temp):
                bin_edges_temp = np.r_[col_min, bin_edges_temp]
            if col_max > max(bin_edges_temp):
                bin_edges_temp = np.r_[bin_edges_temp, col_max]
            bin_edges[jj] = bin_edges_temp

        self.bin_edges_ = bin_edges
        self.n_bins_ = np.array([len(ii) - 1 for ii in bin_edges])

        if 'onehot' in self.encode:
            self._encoder = OneHotEncoder(
                categories=[np.arange(i) for i in self.n_bins_],
                sparse=self.encode == 'onehot')
            # Fit the OneHotEncoder with toy datasets
            # so that it's ready for use after the KBinsDiscretizer is fitted
            self._encoder.fit(np.zeros((1, len(self.n_bins_)), dtype=int))

        return self


    def get_feature_names(self, input_features=None):
        
        if input_features!=None:
            return self._encoder.get_feature_names(input_features=input_features)
        elif self.input_features!=None:
            return self._encoder.get_feature_names(input_features=self.input_features)
        else:
            return self._encoder.get_feature_names(input_features=None)

