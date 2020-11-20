import numpy as np
import pandas as pd

import unittest
from CategoricalCatLimit import CategoricalCatLimit


class TestCategoricalCatLimit(unittest.TestCase):

    def test_strings(self):
        cat_list=[ [ii] for ii in ["huge", "big", "small", "tiny", "big", "small", "tiny"]]
        cat_array = np.array(cat_list)
        cat_df = pd.DataFrame(cat_array)

        
        for X in [cat_list, cat_array, cat_df]:

            # Check nothing is done as default cat_num is 10
            cat_limit00 = CategoricalCatLimit()
            cat_00 = cat_limit00.fit_transform(X)

            self.assertListEqual(cat_limit00.allowed_value[0], ["big", "small", "tiny", "huge"])
            self.assertListEqual(cat_00[0], ["huge", "big", "small", "tiny", "big", "small", "tiny"])

            # Check Capping can be applied
            cat_limit01 = CategoricalCatLimit(cat_num=[3])
            cat_01 = cat_limit01.fit_transform(X)

            self.assertListEqual(cat_limit01.allowed_value[0], ["big", "small", "tiny"])
            self.assertListEqual(cat_01[0], ["other", "big", "small", "tiny", "big", "small", "tiny"])

            # Check default value can be changed
            cat_limit02 = CategoricalCatLimit(cat_num=[3], other_value=["NA"])
            cat_02 = cat_limit02.fit_transform(cat_array)

            self.assertListEqual(cat_limit02.allowed_value[0], ["big", "small", "tiny"])
            self.assertListEqual(cat_02[0], ["NA", "big", "small", "tiny", "big", "small", "tiny"])
            self.assertListEqual(cat_limit02.other_value, ["NA"])

            # Check the level cap can be set using percentage of exposure
            cat_limit03 = CategoricalCatLimit(cat_num=[.15])
            cat_03 = cat_limit03.fit_transform(X)

            self.assertListEqual(cat_limit03.allowed_value[0], ["big", "small", "tiny"])
            self.assertListEqual(cat_03[0], ["other", "big", "small", "tiny", "big", "small", "tiny"])
            self.assertListEqual(cat_limit03.cat_num_, [3])
            self.assertListEqual(cat_limit03.cat_num, [.15])
        
    def test_int(self):
        int_list=[ [ii] for ii in [1,2,3,4,2,3,4]]
        int_array = np.array(int_list)
        int_df = pd.DataFrame(int_array)

        for X in [int_list, int_array, int_df]:

            # Check nothing is done as default cat_num is 10
            int_limit00 = CategoricalCatLimit()
            int_00 = int_limit00.fit_transform(X)

            self.assertListEqual(int_limit00.allowed_value[0], [2., 3., 4., 1.])
            self.assertListEqual(int_00[0], [1.,2.,3.,4.,2.,3.,4.])

            # Check Capping can be applied
            int_limit01 = CategoricalCatLimit(cat_num=[3])
            int_01 = int_limit01.fit_transform(X)

            self.assertListEqual(int_limit01.allowed_value[0], [2, 3, 4])
            self.assertListEqual(int_01[0], [-1.,2.,3.,4.,2.,3.,4.])

            # Check default value can be changed
            int_limit02 = CategoricalCatLimit(cat_num=[3], other_value=[-999])
            int_02 = int_limit02.fit_transform(X)

            self.assertListEqual(int_limit02.allowed_value[0], [2., 3., 4.])
            self.assertListEqual(int_02[0], [-999.,2.,3.,4.,2.,3.,4.])
            self.assertListEqual(int_limit02.other_value, [-999.])

            # Check the level cap can be set using percentage of exposure            
            flt_limit03 = CategoricalCatLimit(cat_num=[.15])
            flt_03 = flt_limit03.fit_transform(X)

            self.assertListEqual(flt_limit03.allowed_value[0], [2., 3., 4.])
            self.assertListEqual(flt_03[0], [-1,2,3,4,2,3,4])
            self.assertListEqual(flt_limit03.cat_num_, [3])
            self.assertListEqual(flt_limit03.cat_num, [.15])

    def test_float(self):
        flt_list=[ [ii] for ii in [1.,2.,3.,4.,2.,3.,4.]]
        flt_array = np.array(flt_list)
        flt_df = pd.DataFrame(flt_array)

        for X in [flt_list, flt_array, flt_df]:

            # Check nothing is done as default cat_num is 10
            flt_limit00 = CategoricalCatLimit()
            flt_00 = flt_limit00.fit_transform(X)

            self.assertListEqual(flt_limit00.allowed_value[0], [2, 3, 4, 1])
            self.assertListEqual(flt_00[0], [1,2,3,4,2,3,4])

            # Check Capping can be applied
            flt_limit01 = CategoricalCatLimit(cat_num=[3])
            flt_01 = flt_limit01.fit_transform(X)

            self.assertListEqual(flt_limit01.allowed_value[0], [2, 3, 4])
            self.assertListEqual(flt_01[0], [-1,2,3,4,2,3,4])

            # Check default value can be changed
            flt_limit02 = CategoricalCatLimit(cat_num=[3], other_value=[-999])
            flt_02 = flt_limit02.fit_transform(X)

            self.assertListEqual(flt_limit02.allowed_value[0], [2, 3, 4])
            self.assertListEqual(flt_02[0], [-999,2,3,4,2,3,4])
            self.assertListEqual(flt_limit02.other_value, [-999])

            # Check the level cap can be set using percentage of exposure
            int_limit03 = CategoricalCatLimit(cat_num=[.15])
            int_03 = int_limit03.fit_transform(X)

            self.assertListEqual(int_limit03.allowed_value[0], [2, 3, 4])
            self.assertListEqual(int_03[0], [-1,2,3,4,2,3,4])
            self.assertListEqual(int_limit03.cat_num_, [3])
            self.assertListEqual(int_limit03.cat_num, [.15])

    def test_multi(self):
        pet_df = np.random.choice(["dog", "cat", "pig", "lama", "elephant"], size=(100,3), p=(0.2, 0.2, 0.2, 0.3, 0.1))

        multi_limit = CategoricalCatLimit(cat_num=[10, 4, 0.15])
        pet_df_t = multi_limit.fit_transform(pet_df)

        self.assertListEqual(sorted(multi_limit.allowed_value[0]), sorted(["dog", "cat", "pig", "lama", "elephant"]))
        self.assertListEqual(sorted(multi_limit.allowed_value[1]), sorted(["dog", "cat", "pig", "lama"]))
        self.assertListEqual(sorted(multi_limit.allowed_value[2]), sorted(["dog", "cat", "pig", "lama"]))

if __name__ == "__main__":
    unittest.main()