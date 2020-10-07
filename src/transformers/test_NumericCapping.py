import numpy as np
import pandas as pd

import unittest
from NumericCapping import NumericCapping


class TestCategoricalCatLimit(unittest.TestCase):
    def test_capping_d1_np(self):
        """test that a d1 numpy array is capped correctly
        """        
        X=np.array([[1.], [2.], [3.], [4.], [5.], [6.], [7.], [8.], [9.], [10.]])

        num_cap = NumericCapping(unk_max=[9], cap_max=[8], cap_min=[3], unk_min=[2])
        X_out = num_cap.fit_transform(X=X)
        
        self.assertListEqual(X_out.flatten().tolist()[1:9], [3.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0])
        self.assertTrue(np.isnan(X_out.flatten().tolist()[0]))
        self.assertTrue(np.isnan(X_out.flatten().tolist()[9]))

    def test_capping_d1_pd(self):
        """test that a d1 pandas dataframe is capped correctly
        """ 
        
        X=np.array([[1.], [2.], [3.], [4.], [5.], [6.], [7.], [8.], [9.], [10.]])
        X_df=pd.DataFrame(X)

        num_cap = NumericCapping(unk_max=[9], cap_max=[8], cap_min=[3], unk_min=[2])
        X_df_out = num_cap.fit_transform(X=X_df)

        
        self.assertListEqual(X_df_out.iloc[:,0].tolist()[1:9], [3.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0])
        self.assertTrue(np.isnan(X_df_out.iloc[0,0]))
        self.assertTrue(np.isnan(X_df_out.iloc[9,0]))
        
    def test_capping_d2_np(self):
        """test that a d2 numpy array is capped correctly
        """ 

        X2=np.array([[1., 10.], [2., 20.], [3., 30.], [4., 40.], [5., 50.], [6., 60.], [7., 70.], [8., 80.], [9., 90.], [10., 100.]])

        num_cap2 = NumericCapping(unk_max=[9, 90], cap_max=[8, 80], cap_min=[3, 30], unk_min=[2, 20])
        X2_out = num_cap2.fit_transform(X=X2)

        self.assertListEqual(X2_out[:,0].flatten().tolist()[1:9], [3.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0])
        self.assertListEqual(X2_out[:,1].flatten().tolist()[1:9], [30.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 80.0])

        for ii in [0,1]:
            for jj in [0,9]:
                self.assertTrue(np.isnan(X2_out[jj,ii]))

    def test_capping_d2_dp(self):
        """test that a d2 pandas dataframe is capped correctly
        """ 

        X2=np.array([[1., 10.], [2., 20.], [3., 30.], [4., 40.], [5., 50.], [6., 60.], [7., 70.], [8., 80.], [9., 90.], [10., 100.]])
        X2_df_out=pd.DataFrame(X2)

        num_cap2 = NumericCapping(unk_max=[9, 90], cap_max=[8, 80], cap_min=[3, 30], unk_min=[2, 20])
        X2_df_out = num_cap2.fit_transform(X=X2)


        self.assertListEqual(X2_df_out[:,0].flatten().tolist()[1:9], [3.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0])
        self.assertListEqual(X2_df_out[:,1].flatten().tolist()[1:9], [30.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 80.0])

        for ii in [0,1]:
            for jj in [0,9]:
                self.assertTrue(np.isnan(X2_df_out[jj, ii]))


if __name__ == "__main__":
    unittest.main()