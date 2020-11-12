import numpy as np
import pandas as pd

import unittest
from SetBinDiscretizer import SetBinDiscretizer


class TestSetBinDiscretizer(unittest.TestCase):

    def test_1D_split(self):
        X = [[-1], [-1], [0], [0], [0], [1], [1], [1], [1]]
        est = SetBinDiscretizer(bin_edges_internal=[[-0.5, 0.5]], encode='ordinal')
        est.fit(X)

        self.assertListEqual(est.transform(X).flatten().tolist(), [0,0,1,1,1,2,2,2,2])

    def test_2D_splits(self):
        # Test it works on 2D data
        X = [[-1, 10], [-1, 9], [0, 8], [0, 7], [0, 6], [1, 5], [1, 4], [1, 3], [1, 2]]
        est = SetBinDiscretizer(bin_edges_internal=[[-0.5, 0.5], [6.5, 5.5, 4.5, -1]], encode='ordinal')
        est.fit(X)

        self.assertListEqual(est.transform(X)[:,0].flatten().tolist(), [0,0,1,1,1,2,2,2,2])
        self.assertListEqual(est.transform(X)[:,1].flatten().tolist(), [3.0, 3.0, 3.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0])

        self.assertListEqual(est.bin_edges_[0].tolist(), [-1., -0.5, 0.5, 1])
        self.assertListEqual(est.bin_edges_[1].tolist(), [-1.0, 4.5, 5.5, 6.5, 10.0])

        # Test Onehot encoder
        est2 = SetBinDiscretizer(bin_edges_internal=[[-0.5, 0.5], [6.5, 5.5, 4.5, -1]], encode='onehot-dense')
        Xt2 = est2.fit_transform(X)

        self.assertListEqual(est2.get_feature_names().tolist(), ['x0_0', 'x0_1', 'x0_2', 'x1_0', 'x1_1', 'x1_2', 'x1_3'])
        self.assertListEqual(est2.get_feature_names(["feat1", "feat2"]).tolist(), ['feat1_0', 'feat1_1', 'feat1_2', 'feat2_0', 'feat2_1', 'feat2_2', 'feat2_3'])

        # Test that you can input feature names
        est3 = SetBinDiscretizer(bin_edges_internal=[[-0.5, 0.5], [6.5, 5.5, 4.5, -1]], encode='onehot', input_features=["feat1", "feat2"])
        Xt3 = est3.fit_transform(X)
        self.assertListEqual(est3.get_feature_names().tolist(), ['feat1_0', 'feat1_1', 'feat1_2', 'feat2_0', 'feat2_1', 'feat2_2', 'feat2_3'])

if __name__ == "__main__":
    unittest.main()