import numpy as np
import pandas as pd

import unittest
from NumericCapping import NumericCapping


class TestCategoricalCatLimit(unittest.TestCase):
    def test_capping_d1(self):
        """test that a d1 numpy array is capped correctly
        """        
        X_flt = np.array([[1.], [2.], [3.], [4.], [5.], [6.], [7.], [8.], [9.], [10.]])
        X_int = np.array([[1.], [2.], [3.], [4.], [5.], [6.], [7.], [8.], [9.], [10.]])
        X_error = np.array([[1., 10.], [2., 20.], [3., 30.], [4., 40.], [5., 50.], [6., 60.], [7., 70.], [8., 80.], [9., 90.], [10., 100.]])

        num_cap = NumericCapping(unk_max=[9], cap_max=[8], cap_min=[3], unk_min=[2])
        X_flt_out = num_cap.fit_transform(X=X_flt)
        
        self.assertListEqual(X_flt_out.flatten().tolist()[1:9], [3.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0]) # Check non-NAs
        self.assertTrue(np.isnan(X_flt_out.flatten().tolist()[0])) # Check NAs
        self.assertTrue(np.isnan(X_flt_out.flatten().tolist()[9])) # Check NAs

        # Check transform rather than fit_transform and check it works on ints
        X_int_out = num_cap.transform(X=X_int)

        self.assertListEqual(X_int_out.flatten().tolist()[1:9], [3.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0]) # Check non-NAs
        self.assertTrue(np.isnan(X_int_out.flatten().tolist()[0])) # Check NAs
        self.assertTrue(np.isnan(X_int_out.flatten().tolist()[9])) # Check NAs

        # Check it works on pandas
        X_df_out = num_cap.transform(X=pd.DataFrame(X_flt))
        
        self.assertListEqual(X_df_out.flatten().tolist()[1:9], [3.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0])
        self.assertTrue(np.isnan(X_df_out.flatten().tolist()[0]))
        self.assertTrue(np.isnan(X_df_out.flatten().tolist()[9]))

        #Check it errors due to transforming different dimensions to fit
        with self.assertRaises(ValueError):
            X_error_out = num_cap.transform(X=X_error)

        #Check it errors due to fitting different dimensions to construction
        with self.assertRaises(ValueError):
            X_error_out = num_cap.fit_transform(X=X_error)


        
    def test_capping_d2(self):
        """test that a d2 numpy array is capped correctly
        """ 

        X2_flt=np.array([[1., 10.], [2., 20.], [3., 30.], [4., 40.], [5., 50.], [6., 60.], [7., 70.], [8., 80.], [9., 90.], [10., 100.]])
        X2_int=np.array([[1 , 10 ], [2 , 20 ], [3 , 30 ], [4 , 40 ], [5 , 50 ], [6 , 60 ], [7 , 70 ], [8 , 80 ], [9 , 90 ], [10 , 100 ]])
        X2_error = np.array([[1.], [2.], [3.], [4.], [5.], [6.], [7.], [8.], [9.], [10.]])

        num_cap2 = NumericCapping(unk_max=[9, 90], cap_max=[8, 80], cap_min=[3, 30], unk_min=[2, 20])
        X2_flt_out = num_cap2.fit_transform(X=X2_flt)

        self.assertListEqual(X2_flt_out[:,0].flatten().tolist()[1:9], [3.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0]) # Check knows
        self.assertListEqual(X2_flt_out[:,1].flatten().tolist()[1:9], [30.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 80.0]) # Check knowns

        #Check unknowns
        for ii in [0,1]:
            for jj in [0,9]:
                self.assertTrue(np.isnan(X2_flt_out[jj,ii]))

        #Check it works on int (use fit_transform to test refit different from 1d test)
        X2_int_out = num_cap2.fit_transform(X=X2_int)

        self.assertListEqual(X2_int_out[:,0].flatten().tolist()[1:9], [3.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0]) # Check knows
        self.assertListEqual(X2_int_out[:,1].flatten().tolist()[1:9], [30.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 80.0]) # Check knowns

        #Check unknowns
        for ii in [0,1]:
            for jj in [0,9]:
                self.assertTrue(np.isnan(X2_int_out[jj,ii]))


        X2_df_out = num_cap2.fit_transform(X=pd.DataFrame(X2_int))
        
        self.assertListEqual(X2_int_out[:,0].flatten().tolist()[1:9], [3.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0]) # Check knows
        self.assertListEqual(X2_int_out[:,1].flatten().tolist()[1:9], [30.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 80.0]) # Check knowns

        #Check unknowns
        for ii in [0,1]:
            for jj in [0,9]:
                self.assertTrue(np.isnan(X2_int_out[jj,ii]))


        #Check it errors due to transforming different dimensions to fit
        with self.assertRaises(ValueError):
            X2_error_out = num_cap2.transform(X=X2_error)

        #Check it errors due to fitting different dimensions to construction
        with self.assertRaises(ValueError):
            X2_error_out = num_cap2.fit_transform(X=X2_error)
        


if __name__ == "__main__":
    unittest.main()