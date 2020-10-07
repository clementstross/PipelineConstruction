import numpy as np
import pandas as pd

import unittest
from CategoricalCatLimit import CategoricalCatLimit


class TestCategoricalCatLimit(unittest.TestCase):

    def test_strings(self):
        cat_df = pd.DataFrame(np.array(["huge", "big", "small", "tiny", "big", "small", "tiny"]), columns=["size"])

        # Check nothing is done as default cat_num is 10
        cat_limit00 = CategoricalCatLimit()
        cat_00 = cat_limit00.fit_transform(cat_df)

        self.assertListEqual(cat_limit00.allowed_value[0], ["big", "small", "tiny", "huge"])
        self.assertListEqual(cat_00["size"].tolist(), ["huge", "big", "small", "tiny", "big", "small", "tiny"])

        # Check Capping can be applied
        cat_limit01 = CategoricalCatLimit(cat_num=[3])
        cat_01 = cat_limit01.fit_transform(cat_df)

        self.assertListEqual(cat_limit01.allowed_value[0], ["big", "small", "tiny"])
        self.assertListEqual(cat_01["size"].tolist(), ["other", "big", "small", "tiny", "big", "small", "tiny"])

        # Check default value can be changed
        cat_limit02 = CategoricalCatLimit(cat_num=[3], other_value=["NA"])
        cat_02 = cat_limit02.fit_transform(cat_df)

        self.assertListEqual(cat_limit02.allowed_value[0], ["big", "small", "tiny"])
        self.assertListEqual(cat_02["size"].tolist(), ["NA", "big", "small", "tiny", "big", "small", "tiny"])

        with self.assertRaises(ValueError):
            cat_03 = CategoricalCatLimit(cat_num=[3,2] , other_value=["NA"])
            cat_03 # stop linting error

        with self.assertRaises(ValueError):
            cat_04 = CategoricalCatLimit(cat_num=[3] , other_value=["NA"])
            cat_df_04 = pd.DataFrame(np.array([["huge", "big", "small", "tiny", "big", "small", "tiny"],["huge", "big", "small", "tiny", "big", "small", "tiny"]]), columns=["size1", "size2"])
            cat_04.fit(cat_df_04)

    def test_int(self):
        cat_df = pd.DataFrame(np.array([1,2,3,4,2,3,4]), columns=["size"])

        # Check nothing is done as default cat_num is 10
        cat_limit00 = CategoricalCatLimit()
        cat_00 = cat_limit00.fit_transform(cat_df)

        self.assertListEqual(cat_limit00.allowed_value[0], [2, 3, 4, 1])
        self.assertListEqual(cat_00["size"].tolist(), [1,2,3,4,2,3,4])

        # Check Capping can be applied
        cat_limit01 = CategoricalCatLimit(cat_num=[3])
        cat_01 = cat_limit01.fit_transform(cat_df)

        self.assertListEqual(cat_limit01.allowed_value[0], [2, 3, 4])
        self.assertListEqual(cat_01["size"].tolist(), [-1,2,3,4,2,3,4])

         # Check default value can be changed
        cat_limit02 = CategoricalCatLimit(cat_num=[3], other_value=[99])
        cat_02 = cat_limit02.fit_transform(cat_df)

        self.assertListEqual(cat_limit02.allowed_value[0], [2, 3, 4])
        self.assertListEqual(cat_02["size"].tolist(), [99,2,3,4,2,3,4]) 


    def test_float(self):
        cat_df = pd.DataFrame(np.array([1.,2.,3.,4.,2.,3.,4.]), columns=["size"])

        # Check nothing is done as default cat_num is 10
        cat_limit00 = CategoricalCatLimit()
        cat_00 = cat_limit00.fit_transform(cat_df)

        self.assertListEqual(cat_limit00.allowed_value[0], [2., 3., 4., 1.])
        self.assertListEqual(cat_00["size"].tolist(), [1.,2.,3.,4.,2.,3.,4.])

        # Check Capping can be applied
        cat_limit01 = CategoricalCatLimit(cat_num=[3])
        cat_01 = cat_limit01.fit_transform(cat_df)

        self.assertListEqual(cat_limit01.allowed_value[0], [2., 3., 4.])
        self.assertListEqual(cat_01["size"].tolist(), [-1.,2.,3.,4.,2.,3.,4.])

         # Check default value can be changed
        cat_limit02 = CategoricalCatLimit(cat_num=[3], other_value=[99])
        cat_02 = cat_limit02.fit_transform(cat_df)

        self.assertListEqual(cat_limit02.allowed_value[0], [2., 3., 4.])
        self.assertListEqual(cat_02["size"].tolist(), [99.,2.,3.,4.,2.,3.,4.])


    def test_multi_col(self):
        d={'size1': ["huge", "big", "small", "tiny", "big", "small", "tiny"], 'size2': [1,2,3,4,2,3,4], 'size3': [1.,2.,3.,4.,2.,3.,4.]}
        cat_df=pd.DataFrame(data=d)

        cat_limit00 = CategoricalCatLimit(cat_num=[3, 1, 10])
        cat_00 = cat_limit00.fit_transform(cat_df)

        self.assertListEqual(cat_limit00.allowed_value[0], ["big", "small", "tiny"])
        self.assertListEqual(cat_limit00.allowed_value[1], [2])
        self.assertListEqual(cat_limit00.allowed_value[2], [2., 3., 4., 1.])

        self.assertListEqual(cat_00["size1"].tolist(), ["other", "big", "small", "tiny", "big", "small", "tiny"])
        self.assertListEqual(cat_00["size2"].tolist(), [-1,2,-1,-1,2,-1,-1])
        self.assertListEqual(cat_00["size3"].tolist(), [1.,2.,3.,4.,2.,3.,4.])

if __name__ == "__main__":
    unittest.main()